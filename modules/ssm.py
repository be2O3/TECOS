from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat



@dataclass
class InitStrategy(object):
    """As SSMs are potentially sensible to initialization, wrap the init strategy"""
    A: str = 'hippo'  # {hippo, random, constant}
    B: str = 'constant'  # {hippo, random, constant}
    C: str = None  # doesn't matter that much
    delta: str = None  # TODO: explore initialization schemes here

    def _init_A(self, log_nA):
        if self.A == 'hippo':
            size = log_nA.size()
            if len(size) == 2:
                d_input, d_state = size
                log_nA.fill_(0).add_(
                    # torch.tile(
                    #     torch.log(torch.arange(1, self.d_state + 1) + 1).view(1, -1),
                    #     [self.d_input, 1]
                    # )
                    repeat(
                        torch.log(torch.arange(1, d_state + 1) + 1),
                        'd -> b d',
                        b=d_input
                    )
                )
            else:
                d_state = size[0]
                log_nA.fill_(0).add_(
                    torch.log(torch.arange(1, d_state + 1) + 1))
        elif self.A == 'random':
            nn.init.xavier_uniform_(log_nA)
        else:
            nn.init.constant_(log_nA, np.log(0.5))

    def _init_B(self, B):
        if self.B == 'hippo':
            d_input, d_state = B.size()
            B.fill_(0).add_(
                repeat(
                    torch.sqrt(2 * torch.arange(1, d_state + 1) + 1),
                    'd -> b d',
                    b=d_input
                )
            )
        elif self.B == 'constant':
            nn.init.constant_(B, 1.)
        else:
            nn.init.xavier_uniform_(B)

    def _init_C(self, C):
        nn.init.xavier_uniform_(C)

    def _init_delta(self, delta: nn.Linear):
        nn.init.constant_(delta.weight, 0.)
        nn.init.constant_(
            delta.bias,
            np.log(np.exp(np.random.uniform(0.001, 0.1)) - 1)
        )

    @torch.no_grad()
    def init(self, log_nA, B, C):
        self._init_A(log_nA)
        self._init_B(B)
        self._init_C(C)