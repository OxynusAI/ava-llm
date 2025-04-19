import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from ava.config import AvaConfig
from ava.utils import apply_rotary_pos_emb


class AvaAttention(nn.Module):
    """Attention module"""

    def __init__(self, config: AvaConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_heads = config.kv_heads if hasattr(config, 'kv_heads') else config.num_attention_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)


    def forward_with_sliding_window(self, 
                hidden_states, 
                attention_mask=None, 
                past_key_value=None, 
                output_attentions=False, 
                use_cache=False, 
                rotary_emb=None,
                position_ids=None,
                window_size=1024
        ):
        
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.kv_heads, self.head_dim
        ).transpose(1, 2) 
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        if rotary_emb is not None:
            cur_seq_len = key_states.shape[2]
            cos, sin = rotary_emb(value_states, seq_len=cur_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if self.kv_heads < self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            value_states = value_states.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        
        context_layer = torch.zeros_like(query_states)
        
        for i in range(seq_length):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_length, i + window_size // 2)
            window_key_states = key_states[:, :, window_start:window_end, :]
            window_value_states = value_states[:, :, window_start:window_end, :]
            
            attention_scores = torch.matmul(
                query_states[:, :, i:i+1, :], 
                window_key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                window_attention_mask = attention_mask[:, :, i:i+1, window_start:window_end]
                attention_scores = attention_scores + window_attention_mask
            
            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_probs = self.attention_dropout(attention_probs)
            window_context = torch.matmul(attention_probs, window_value_states)
            context_layer[:, :, i:i+1, :] = window_context
        
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        
        output = self.o_proj(context_layer)
        outputs = (output, past_key_value)

        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs

        
    def forward(self, 
                hidden_states, 
                attention_mask = None, 
                past_key_value = None, 
                output_attentions = False, 
                use_cache         = False, 
                rotary_emb        = None,
                position_ids      = None
        ):
        
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.kv_heads, self.head_dim
        ).transpose(1, 2) 
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        if rotary_emb is not None:
            cur_seq_len = key_states.shape[2]
            cos, sin = rotary_emb(value_states, seq_len=cur_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if self.kv_heads < self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            value_states = value_states.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        
        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_probs = self.attention_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        
        output = self.o_proj(context_layer)
        outputs = (output, past_key_value)

        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs