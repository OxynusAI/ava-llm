import torch

from torch import nn
from typing import Tuple

from core import AvaConfig
from core.layers import AvaAttention, AvaMLP, RMSNorm

class AvaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: AvaConfig,
    ):
        super().__init__()
        self.self_attn = AvaAttention(
            hidden_size  = config.hidden_size,
            num_heads    = config.num_attention_heads,
            num_kv_heads = config.num_key_value_heads,
            head_dim     = config.head_dim,
            quant        = config.quant,
        )
        
        self.mlp = AvaMLP(
            hidden_size       = config.hidden_size,
            intermediate_size = config.intermediate_size,
            quant             = config.quant,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states    = hidden_states,
            freqs_cis        = freqs_cis,
            kv_write_indices = kv_write_indices,
            kv_cache         = kv_cache,
            mask             = mask,
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
