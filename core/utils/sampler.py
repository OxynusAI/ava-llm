import torch
from torch import nn

from typing import Optional, Union

class AvaSampler(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(
        self,
        embedding_matrix: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_probabilities: torch.Tensor,
        top_k_values: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding_matrix.t())
        
        if embedding_bias is not None:
            logits += embedding_bias

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)

        logits.div_(temperatures.unsqueeze(dim=1))
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float)
        sorted_probs, sorted_probs_indices = torch.sort(probabilities, dim=-1, descending=True)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_probabilities_mask = (cumulative_probs - sorted_probs) > top_probabilities.unsqueeze(dim=1)
        sorted_probs = torch.where(top_probabilities_mask, 0, sorted_probs)

        top_k_values_mask = torch.arange(sorted_probs_indices.shape[-1], device=sorted_probs_indices.device)
        top_k_values_mask = top_k_values_mask.expand(sorted_probs_indices.shape[0], -1)
        top_k_values_mask = top_k_values_mask >= top_k_values.unsqueeze(dim=1)
        sorted_probs = torch.where(top_k_values_mask, 0, sorted_probs)

        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        probabilities = torch.gather(
            sorted_probs, 
            dim = -1, 
            index = torch.argsort(
                sorted_probs_indices, 
                dim = -1
            )
        )

        return torch.multinomial(
            probabilities, 
            num_samples = 1, 
            replacement = True
        ).squeeze(dim=-1)
