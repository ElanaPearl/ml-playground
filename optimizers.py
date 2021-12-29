from typing import Dict, List

import torch
from torch.functional import Tensor


class Adam:
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=10e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.timestep = 0

        self.first_moments: Dict[torch.nn.Parameter, Tensor] = {}
        self.second_moments: Dict[torch.nn.Parameter, Tensor] = {}
        for param in params:
            self.first_moments[param] = torch.zeros_like(param)
            self.second_moments[param] = torch.zeros_like(param)

    def step(self):
        self.timestep += 1
        beta_1 = self.beta_1
        beta_2 = self.beta_2

        def add_to_moving_average(old, new, beta):
            return old * beta + new * (1 - beta)

        for p in self.params:
            first_moment = add_to_moving_average(
                old=self.first_moments[p], new=p.grad, beta=beta_1
            )
            second_moment = add_to_moving_average(
                old=self.second_moments[p], new=p.grad ** 2, beta=beta_2
            )
            bias_corrected_first_moment = first_moment * (1 - beta_1 ** self.timestep)
            bias_corrected_second_moment = second_moment * (1 - beta_2 ** self.timestep)
            p.data -= (
                self.lr
                * bias_corrected_first_moment
                / (torch.sqrt(bias_corrected_second_moment) + self.eps)
            )
            self.first_moments[p] = first_moment
            self.second_moments[p] = second_moment
