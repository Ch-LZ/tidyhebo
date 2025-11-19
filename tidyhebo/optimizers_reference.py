"""
Оптимизаторы для сравнения с HEBO
"""
import numpy as np
import torch
from typing import List

from tidyhebo.common import dtype

# region Real
class RealParameter:
    def __init__(self, index, lb, ub):
        self.index = index
        self.lb = lb
        self.ub = ub


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

# todo TuRBO

# todo RayTune (vanilla HEBO)