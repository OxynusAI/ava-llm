import torch

def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    
    frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    time_steps = torch.arange(end, device=frequencies.device)
    freqs_cis = torch.polar(torch.ones_like(time_steps), torch.outer(time_steps, frequencies).float()) 
    
    return freqs_cis


def apply_rotary_emb(input_tensor: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    complex_input = torch.view_as_complex(
        torch.stack(
            torch.chunk(
                input_tensor.transpose(1, 2).float(), 
                2, 
                dim=-1
            ),
            dim=-1
        )
    )
    
    rotated_input = torch.view_as_real(complex_input * freqs_cis).type_as(input_tensor)
    rotated_input = torch.cat(torch.chunk(rotated_input, 2, dim=-1), dim=-2)
    
    return rotated_input.reshape(
        rotated_input.shape[0], 
        rotated_input.shape[1], 
        rotated_input.shape[2],
        -1
    ).transpose(1, 2)
