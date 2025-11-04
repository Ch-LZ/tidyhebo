"""
Оптимизаторы для сравнения с HEBO
"""
import numpy as np
import torch
from typing import List

from tidyhebo.common import dtype

# region Real
class RealParameter:
    def __init__(self, index, lb, ub, name=None):
        self.index = index
        self.lb = lb
        self.ub = ub
        self.name=name


class RealDesignSpace:
    def __init__(self, parameters: List[RealParameter]):
        self.lb_all = torch.tensor([p.lb for p in parameters], dtype=dtype)
        self.ub_all = torch.tensor([p.ub for p in parameters], dtype=dtype)
        self.order = [p.index for p in parameters]
        assert not (np.array(range(len(parameters))) - np.array(self.order)).any()

    
    def to_opt_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        x - 2D (n rows, dim cols)
        """
        return (x - self.lb_all) / (self.ub_all - self.lb_all)
    

    def to_domain_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        x - 2D (n rows, dim cols)
        """
        return x * (self.ub_all - self.lb_all) +  self.lb_all

# endregion


#* Sobol
class SobolOptimizer:
    def __init__(self, parameters: List[RealParameter], train_x=None, train_y=None):
        from scipy.stats.qmc import Sobol
        self.design_space: RealDesignSpace = RealDesignSpace(parameters)  # bounds, parameters
        self.model = None
        self.train_y = train_y
        self.train_x = train_x
        self.sampler = Sobol(self.train_x.shape[-1])


    def suggest(self) -> torch.Tensor:
        candidates = torch.from_numpy(self.sampler.random(1))
        candidates = self.design_space.to_domain_space(candidates)
        return candidates


    def observe(self, x, y):
        if y.dim() == 1:
            y = y.reshape(-1, 1)  # column
        x = self.design_space.to_opt_space(x)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)


#* random
class RandomOptimizer:
    def __init__(self, parameters: List[RealParameter], train_x=None, train_y=None):

        self.design_space: RealDesignSpace = RealDesignSpace(parameters)  # bounds, parameters
        self.model = None
        self.train_y = train_y
        self.train_x = train_x


    def suggest(self) -> torch.Tensor:
        sample = torch.rand(1, self.train_x.shape[-1])
        # candidates = torch.from_numpy(sample)
        candidates = self.design_space.to_domain_space(sample)
        return candidates


    def observe(self, x, y):
        if y.dim() == 1:
            y = y.reshape(-1, 1)  # column
        x = self.design_space.to_opt_space(x)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)


import optuna

#* Optuna
class OptunaOptimizer:
    def __init__(self, parameters: List[RealParameter], train_x=None, train_y=None):
        self.parameters = parameters
        self.study = optuna.create_study(direction="maximize")
        self._prev_trial = None

    def suggest(self) -> torch.Tensor:
        trial = self.study.ask()
        self._prev_trial = trial
        cand = []
        for i, p in enumerate(self.parameters):
            x = trial.suggest_float(f"{i}", p.lb, p.ub)
            cand.append(x)
        return torch.tensor(cand, dtype=dtype).reshape(1, -1)


    def observe(self, x, y) -> None:
        y = y.reshape(-1).item()
        self.study.tell(self._prev_trial, y)
        self._prev_trial = None


#* qLogEI
from tidyhebo.base_model import SimpleModel
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

class qLogEIOptimizer:
    """
    Стереотипный сетап GP/EI без намеренно сделаный без трансформов но с численно-устойчивой acquisition function и достаточно большим количество семплов в Sobol sampler чтобы избежать численных проблем.
    """
    def __init__(self, parameters: List[RealParameter], train_x=None, train_y=None):
        self.design_space: RealDesignSpace = RealDesignSpace(parameters)  # bounds, parameters
        dim = len(parameters)
        train_x = torch.empty((0, dim)) if train_x is None else train_x
        train_y = torch.empty((0, 1)) if train_y is None else train_y
        self.train_x = self.design_space.to_opt_space(train_x)  # хранятся в пространстве оптимизатора
        self.train_y = train_y
        # model config
        self.kernel = "default"
        self.power_outcome = False
        self.warp_input = False
        self.state_dict = None


    def suggest(self, q=1):
        model: SimpleModel = SimpleModel(
            train_x=self.train_x,
            train_y=self.train_y,
            kernel=self.kernel,
            power_outcome=self.power_outcome,
            warp_input=self.warp_input,
        )
        # update model
        model.train_model(state_dict=self.state_dict)
        # save to future runs
        self.state_dict = model.model.state_dict()
        # make acquisition function
        samplint_size = 512 * 4 * 2
        acqf = qLogExpectedImprovement(
            model=model.model,
            best_f=self.train_y.max().item(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([samplint_size]))
        )
        # optimizer acquisition
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=model.bounds,
            q=1,
            num_restarts=10,
            raw_samples=min(2000, 500 * self.train_x.shape[-1]),
        )
        # transform into space
        res = self.design_space.to_domain_space(candidates)
        return res


    def observe(self, X, y) -> None:
        """
        Получает наблюдения в пространстве дизайна
        """
        # add new y observation to set
        if y.dim() == 1:
            y = y.reshape(-1, 1)  # column
        x = self.design_space.to_opt_space(X)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)


#* GIBBON
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

class GibbonOptimizer:
    def __init__(self, parameters: List[RealParameter], train_x=None, train_y=None):
        self.design_space: RealDesignSpace = RealDesignSpace(parameters)  # bounds, parameters
        dim = len(parameters)
        train_x = torch.empty((0, dim)) if train_x is None else train_x
        train_y = torch.empty((0, 1)) if train_y is None else train_y
        self.train_x = self.design_space.to_opt_space(train_x)  # хранятся в пространстве оптимизатора
        self.train_y = train_y
        # model config
        self.kernel = "Matern52_Gamma"
        self.power_outcome = True
        self.warp_input = True
        self.state_dict = None
        self.model = None

    def suggest(self):
        model: SimpleModel = SimpleModel(
            train_x=self.train_x,
            train_y=self.train_y,
            kernel=self.kernel,
            power_outcome=self.power_outcome,
            warp_input=self.warp_input,
        )
        # update model
        model.train_model(state_dict=self.state_dict)
        # save to future runs
        self.state_dict = model.model.state_dict()
        self.model = model
        # make acquisition function
        n_candidates = 2048
        sampler = torch.quasirandom.SobolEngine(self.train_x.shape[-1], scramble=True)
        candidate_set = sampler.draw(n_candidates)
        qGIBBON = qLowerBoundMaxValueEntropy(model.model, candidate_set)
        # optimize acquisition
        candidates, _ = optimize_acqf(
            acq_function=qGIBBON,
            bounds=model.bounds,
            q=1,
            num_restarts=10,
            raw_samples=n_candidates,
        )
        # transform into space
        res = self.design_space.to_domain_space(candidates)
        return res


    def observe(self, X, y) -> None:
        """
        Получает наблюдения в пространстве дизайна
        """
        # add new y observation to set
        if y.dim() == 1:
            y = y.reshape(-1, 1)  # column
        x = self.design_space.to_opt_space(X)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)

# region models
import torch

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Warp
from gpytorch.priors.torch_priors import LogNormalPrior

from .common import dtype, device, GP_Tensor
from .yeo_johnson_warp import get_closure, YeoJohnsonGPytorch


class YJModel:
    def __init__(self, 
                 train_x: GP_Tensor,
                 train_y,
                 use_input_warp=False):
        self.model: SingleTaskGP = None
        self.train_y = train_y
        self.train_x = train_x
        self.mll = None

        self.outcome_transform = None

        n_x = self.train_x.shape[1]
        self.bounds = torch.repeat_interleave(
            torch.tensor([[0,1],], device=device, dtype=dtype), n_x, dim=0).T
        
        self.use_input_warp = use_input_warp
    

    def train_model(self, state_dict=None):

        input_warp = None
        if self.use_input_warp:
            input_warp = Warp(  # уже содержит внутри kumaraswamy преобразование (и нормализацию)
                d=self.train_x.shape[-1],
                indices=list(range(self.train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
                bounds=self.bounds,
            )

        self.model = SingleTaskGP(
            train_X=self.train_x,
            train_Y=self.train_y,
            input_transform=input_warp,
        ).to(self.train_x)
        
        self.outcome_transform = YeoJohnsonGPytorch()
        self.model.outcome_warp = self.outcome_transform  # нужно положить его как поле, чтобы потом он был найден в computational графе

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
        
        fit_gpytorch_mll(self.mll, 
                    closure=get_closure(self.mll, self.model.outcome_warp),
                            max_attempts=10,   # at this momemt the model fails to optimize unexpectedly.
                            )


class SepYJModel:
    def __init__(self,
                 train_x: GP_Tensor,
                 train_y):
        
        self.model: SingleTaskGP = None
        self.train_y = train_y
        self.train_x = train_x
        self.mll = None

        self.outcome_transform = None
    
        n_x = self.train_x.shape[1]
        self.bounds = torch.repeat_interleave(
            torch.tensor([[0,1],], device=device, dtype=dtype), n_x, dim=0).T


    def train_model(self, state_dict=None):
        self.outcome_transform = YeoJohnsonGPytorch()
        self.outcome_transform._optimize(self.train_y)
    
        with torch.no_grad():
            y_transformed = self.outcome_transform(self.train_y)

        self.model = SingleTaskGP(
            train_X=self.train_x,
            train_Y=y_transformed,
        ).to(self.train_x)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
        
        fit_gpytorch_mll(self.mll)


class SepYJKumModel:
    def __init__(self,
                 train_x: GP_Tensor,
                 train_y):
        
        self.model: SingleTaskGP = None
        self.train_y = train_y
        self.train_x = train_x
        self.mll = None

        self.outcome_transform = None
    
        n_x = self.train_x.shape[1]
        self.bounds = torch.repeat_interleave(
            torch.tensor([[0,1],], device=device, dtype=dtype), n_x, dim=0).T


    def train_model(self, state_dict=None):
        self.outcome_transform = YeoJohnsonGPytorch()
        self.outcome_transform._optimize(self.train_y)

        input_warp = Warp(  # уже содержит внутри kumaraswamy преобразование (и нормализацию)
                d=self.train_x.shape[-1],
                indices=list(range(self.train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
                bounds=self.bounds,
            )

        with torch.no_grad():
            y_transformed = self.outcome_transform(self.train_y)

        self.model = SingleTaskGP(
            train_X=self.train_x,
            train_Y=y_transformed,
            input_transform=input_warp,
        ).to(self.train_x)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
        
        fit_gpytorch_mll(self.mll)
# endregion