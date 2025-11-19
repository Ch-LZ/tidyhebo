import torch

from botorch.acquisition import (
    AcquisitionFunction,
    qLogNoisyExpectedImprovement, 
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
    ):
        super().__init__(model)
        self.model = model

        # Use MC for better numeric stability and robustness.
        samplint_size = 512 * 4 * 2
        # разные семплеры - обязательны. Обратное - вызывает ошибку в qLogNoisyExpectedImprovement
        self.qLogNEI = qLogNoisyExpectedImprovement(model, X_baseline=train_x, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([samplint_size])))
        self.qPI = qProbabilityOfImprovement(model, best_f=best_f, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([samplint_size])))
        self.q_UCB = qUpperConfidenceBound(model, beta=beta, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([samplint_size])))
        self.n_obj: int = 3


    def forward(self, X: Tensor) -> Tensor:
        # Вычисление значений функций (автоматическое распознавание batch_shape)
        val_ei = torch.exp(self.qLogNEI(X))
        val_pi = self.qPI(X)
        val_ucb = self.q_UCB(X)
        return torch.column_stack((val_ei, val_pi, val_ucb))
