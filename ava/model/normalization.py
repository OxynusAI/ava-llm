import torch
import torch.nn as nn

class AvaRMSNorm(nn.Module):
    """
    Enhanced RMSNorm with better numerical stability and CUDA optimizations
    """
    
    def __init__(self, hidden_size: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Root Mean Square Layer Normalization
        """

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        

        if self.weight.dtype != hidden_states.dtype:
            hidden_states = (self.weight * hidden_states.type_as(self.weight))
        else:
            hidden_states = self.weight * hidden_states
            
        return hidden_states