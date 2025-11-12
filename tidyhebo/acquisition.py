import torch

from botorch.acquisition import (
    AcquisitionFunction,
    qLogNoisyExpectedImprovement,
    qLogExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound
    )
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from .common import GP_Tensor


class HedgedPareto(AcquisitionFunction):
    """
    Обновленный и улучшенный вариант HEBO (https://doi.org/10.1613/jair.1.13643)
    """
    def __init__(
        self,
        model: SingleTaskGP,
        best_f: float,
        beta: float,
        train_x: GP_Tensor,
        use_logei=False,  # use qLogEI instead of qLogNEI. qLogNEI just harder to calculate imho.
        objective=None,
    ):
        super().__init__(model)
        self.model = model

        # Use MC for better numeric stability and robustness.
        sampl_int_size = int(512 * 4 * 2 * 2.5)
        # разные семплеры - обязательны. Обратное - вызывает ошибку в qLogNoisyExpectedImprovement
        if not use_logei:
            self.qLogNEI = qLogNoisyExpectedImprovement(model, X_baseline=train_x, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([sampl_int_size])), objective=objective)
        else:
            self.qLogNEI = qLogExpectedImprovement(model, best_f=best_f, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([sampl_int_size])), objective=objective)
        self.qPI = qProbabilityOfImprovement(model, best_f=best_f, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([sampl_int_size])), objective=objective)
        self.q_UCB = qUpperConfidenceBound(model, beta=beta, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([sampl_int_size])), objective=objective)
        self.n_obj: int = 3


    def forward(self, X: Tensor) -> Tensor:
        # Вычисление значений функций (автоматическое распознавание batch_shape)
        val_ei = torch.exp(self.qLogNEI(X))
        val_pi = self.qPI(X)
        val_ucb = self.q_UCB(X)
        return torch.column_stack((val_ei, val_pi, val_ucb))
