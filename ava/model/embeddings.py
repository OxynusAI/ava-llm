import torch
import torch.nn as nn

from typing import Optional, Tuple

class AvaRotaryEmbedding(nn.Module):
    """
    Enhanced rotary positional embeddings with frequency scaling and improved caching
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        

        self._precomputed_freqs = {}
        for seq_len in [512, 1024, 2048, 4096, 8192]:
            if seq_len <= max_position_embeddings:
                self._precomputed_freqs[seq_len] = self._compute_freqs(seq_len)
    

    def _compute_freqs(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = min(seq_len, self.max_position_embeddings)
        t = torch.arange(seq_len, device=self.inv_freq.device)

        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()


        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of rotary embeddings
        
        Args:
            x: Input tensor
            seq_len: Optional sequence length to compute embeddings for
            
        Returns:
            Tuple of cosine and sine embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        if seq_len in self._precomputed_freqs:
            return self._precomputed_freqs[seq_len]
        

        cos_cached, sin_cached = self._compute_freqs(seq_len)
        return cos_cached, sin_cached

