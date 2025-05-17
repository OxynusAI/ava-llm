from typing import Optional, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized rotary position embeddings application that handles different sequence lengths for q and k.
    This version is heavily optimized for speed using vectorized operations.
    """

    batch_size_q, seq_len_q, num_heads_q, head_dim = q.shape
    batch_size_k, seq_len_k, num_heads_k, head_dim_k = k.shape
    
    if position_ids is not None and position_ids.shape[0] == batch_size_q:
        q_position_ids = position_ids[:, :seq_len_q] if position_ids.shape[1] >= seq_len_q else position_ids
        k_position_ids = position_ids[:, :seq_len_k] if position_ids.shape[1] >= seq_len_k else position_ids
    else:

        q_position_ids = None
        k_position_ids = None


    def create_embeddings(seq_len, pos_ids=None, tensor_device=None):
        if pos_ids is None:
            emb_cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
            emb_sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]

            emb_cos = emb_cos.expand(batch_size_q, -1, -1, -1)
            emb_sin = emb_sin.expand(batch_size_q, -1, -1, -1) 
        else:
            device = tensor_device if tensor_device is not None else pos_ids.device
            
            max_pos = cos.shape[0] - 1
            safe_pos_ids = torch.clamp(pos_ids, 0, max_pos)

            flat_pos_ids = safe_pos_ids.reshape(-1)  
            selected_cos = cos.index_select(0, flat_pos_ids) 
            selected_sin = sin.index_select(0, flat_pos_ids)

            emb_cos = selected_cos.reshape(batch_size_q, -1, 1, head_dim)
            emb_sin = selected_sin.reshape(batch_size_q, -1, 1, head_dim)

            if emb_cos.shape[1] > seq_len:
                emb_cos = emb_cos[:, :seq_len]
                emb_sin = emb_sin[:, :seq_len]
        
        return emb_cos, emb_sin


    q_cos, q_sin = create_embeddings(seq_len_q, q_position_ids, q.device)
    k_cos, k_sin = create_embeddings(seq_len_k, k_position_ids, k.device)
    

    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    return torch.cat((-x2, x1), dim=-1)

class AvaAttention(nn.Module):
    """
    Enhanced attention module with grouped-query attention, 
    flash attention support, and optimized KV cache handling
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        

        self.kv_heads = config.kv_heads if hasattr(config, "kv_heads") else config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self._use_flash_attention = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        rotary_emb: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length, _ = hidden_states.shape
        

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        

        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(1, 2)
        

        kv_seq_len = seq_length
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        

        if rotary_emb is not None:
            cos, sin = rotary_emb(query_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        

        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
            value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)
        

        if self._use_flash_attention and attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, 
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
            
            attention_probs = None  # Not available with flash attention
        else:

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            

            attn_output = torch.matmul(attn_weights, value_states)
            attention_probs = attn_weights if output_attentions else None
        

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        

        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if use_cache:
            outputs += (past_key_value,)
        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs
