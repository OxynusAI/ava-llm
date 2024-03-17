import torch

from torch import nn

class RMSNorm(nn.Module):

    def __init__(
        self,
        dimension: int,
        epsilon: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.add_unit_offset = add_unit_offset
        self.scaling_factor = nn.Parameter(torch.zeros(dimension))

    def _normalize(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x):
        x = self._normalize(x.float()).type_as(x)
        
        return x * (1 + self.scaling_factor) if self.add_unit_offset else x * self.scaling_factor
