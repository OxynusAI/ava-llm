import torch
import torch.nn.functional as F

from torch import nn

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, is_quantized: bool):
        super().__init__()
        
        if is_quantized:
            self.weight = nn.Parameter(
                torch.empty(
                    (
                        out_features, 
                        in_features
                    ), 
                    dtype=torch.int8
                ),
                
                requires_grad=False,
            )
            
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
            
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (
                        out_features, 
                        in_features
                    )
                ),
                
                requires_grad=False,
            )
            
        self.is_quantized = is_quantized

    def forward(self, x):
        weight = self.weight
        
        if self.is_quantized:
            weight = weight * self.weight_scaler.unsqueeze(-1)
            
        return F.linear(x, weight)
