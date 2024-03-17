import torch
import torch.nn.functional as F

from torch import nn

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        is_quantized: bool
    ):
        super().__init__()
    
        if is_quantized:
            self.weight = nn.Parameter(
                torch.empty(
                    (
                        num_embeddings, 
                        embedding_dim
                    ), 
                    
                    dtype=torch.int8
                ),
                
                requires_grad=False,
            )
            
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
            
        self.is_quantized = is_quantized

    def forward(self, x):
        weight = self.weight
        
        if self.is_quantized:
            weight = weight * self.weight_scaler.unsqueeze(-1)
            
        return F.embedding(x, weight)
