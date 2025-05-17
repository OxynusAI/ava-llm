import torch
import torch.nn as nn
import torch.nn.functional as F

class AvaMLP(nn.Module):
    """
    Enhanced MLP with SwiGLU activation and optimized computation
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using SwiGLU activation
        """

        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        

        intermediate_output = gate_output * up_output
        

        return self.down_proj(intermediate_output)


# import math
# from typing import Optional, Union, Literal

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import bitsandbytes as bnb

# class AvaMLP(nn.Module):
#     """
#     Optimized MLP block for Ava models with SwiGLU activation.
    
#     This implementation uses several optimizations:
#     - Native SiLU/SwiGLU activation for better performance
#     - Optional bias configuration
#     - Flexible activation function selection
#     - Flash attention compatibility
#     - Efficient memory layout
#     - Support for quantization and tensor parallelism
#     """
    
#     def __init__(
#         self, 
#         config,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#         use_bias: bool = False,
#         use_memory_efficient_linear: bool = True,
#         activation: str = "silu"
#     ):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
        
#         # Handle activation function selection
#         self.activation = activation.lower() if isinstance(activation, str) else config.hidden_act
        
#         # Better initialization for improved training stability
#         # Use scaled initialization based on network depth
#         init_std = config.initializer_range / math.sqrt(2 * config.num_hidden_layers)
        
#         # Optimize linear layers
#         linear_cls = nn.Linear
#         if use_memory_efficient_linear and hasattr(nn.utils, 'skip_init'):
#             # Use memory efficient initialization if available
#             linear_cls = lambda *args, **kwargs: nn.utils.skip_init(nn.Linear, *args, **kwargs)
            
#         factory_kwargs = {'device': device, 'dtype': dtype}
        
#         # Create projection layers
#         self.gate_proj = linear_cls(
#             self.hidden_size, 
#             self.intermediate_size, 
#             bias=use_bias,
#             **factory_kwargs
#         )
        
#         self.up_proj = linear_cls(
#             self.hidden_size, 
#             self.intermediate_size, 
#             bias=use_bias,
#             **factory_kwargs
#         )
        
#         self.down_proj = linear_cls(
#             self.intermediate_size, 
#             self.hidden_size, 
#             bias=use_bias,
#             **factory_kwargs
#         )
        
#         # Initialize weights for better convergence
#         self._init_weights(init_std)
        
#     def _init_weights(self, std: float):
#         """Initialize weights with improved distribution."""
#         nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
#         nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
#         # Use smaller initialization for the output projection
#         nn.init.normal_(self.down_proj.weight, mean=0.0, std=std/math.sqrt(2))
        
#         # Initialize bias if present
#         if hasattr(self.gate_proj, 'bias') and self.gate_proj.bias is not None:
#             nn.init.zeros_(self.gate_proj.bias)
#             nn.init.zeros_(self.up_proj.bias)
#             nn.init.zeros_(self.down_proj.bias)
    
#     def _get_activation_fn(self, x: torch.Tensor) -> torch.Tensor:
#         """Apply the selected activation function."""
#         if self.activation == "silu" or self.activation == "swish":
#             # Use native implementation when available (PyTorch >= 1.7)
#             if hasattr(F, 'silu'):
#                 return F.silu(x)
#             else:
#                 return x * torch.sigmoid(x)
#         elif self.activation == "gelu":
#             return F.gelu(x)
#         elif self.activation == "gelu_new":
#             return F.gelu(x, approximate="tanh")
#         elif self.activation == "relu":
#             return F.relu(x)
#         else:
#             # Default to SiLU as in the original implementation
#             return x * torch.sigmoid(x)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass with optimized computation pattern.
        
#         Args:
#             x: Input tensor of shape [batch_size, seq_length, hidden_size]
            
#         Returns:
#             Output tensor of shape [batch_size, seq_length, hidden_size]
#         """
#         # Compute gate and up projections in parallel
#         if torch.jit.is_scripting() or torch.jit.is_tracing():
#             gate = self._get_activation_fn(self.gate_proj(x))
#             up = self.up_proj(x)
#             return self.down_proj(gate * up)
#         else:
#             # More efficient fused computation using SwiGLU pattern
#             hidden_states = self.down_proj(
#                 self._get_activation_fn(self.gate_proj(x)) * self.up_proj(x)
#             )
#             return hidden_states


# class AvaMLPQuantized(nn.Module):
#     """
#     Memory-efficient MLP implementation with support for quantization.
#     Uses 4/8-bit weight quantization for reduced memory footprint.
#     """
    
#     def __init__(
#         self,
#         config,
#         quantize: Literal['4bit', '8bit', 'none'] = 'none',
#         device: Optional[torch.device] = None
#     ):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
        
#         # Dynamic import of quantization modules if needed
#         if quantize != 'none' and not torch.jit.is_scripting():
#             try:
#                 if quantize == '4bit':
#                     linear_cls = bnb.nn.Linear4bit
#                     compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
#                     compress_statistics = True
#                     quant_type = 'nf4'
#                 elif quantize == '8bit':
#                     linear_cls = bnb.nn.Linear8bitLt
#                     compute_dtype = None
#                     compress_statistics = False
#                     quant_type = None
                
#                 factory_kwargs = {
#                     'has_fp16_weights': False,
#                     'threshold': 6.0,
#                     'compute_dtype': compute_dtype,
#                     'compress_statistics': compress_statistics,
#                     'quant_type': quant_type
#                 }
                
#                 # Filter out None values
#                 factory_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
                
#                 # Create quantized layers
#                 self.gate_proj = linear_cls(
#                     self.hidden_size, 
#                     self.intermediate_size,
#                     bias=False,
#                     **factory_kwargs
#                 )
                
#                 self.up_proj = linear_cls(
#                     self.hidden_size, 
#                     self.intermediate_size,
#                     bias=False,
#                     **factory_kwargs
#                 )
                
#                 self.down_proj = linear_cls(
#                     self.intermediate_size, 
#                     self.hidden_size,
#                     bias=False,
#                     **factory_kwargs
#                 )
                
#             except ImportError:
#                 print("Warning: bitsandbytes not installed. Using standard linear layers.")
#                 self._setup_standard_layers(device)
#         else:
#             self._setup_standard_layers(device)
        
#     def _setup_standard_layers(self, device):
#         """Setup standard linear layers when quantization is not available."""
#         self.gate_proj = nn.Linear(
#             self.hidden_size, 
#             self.intermediate_size, 
#             bias=False,
#             device=device
#         )
        
#         self.up_proj = nn.Linear(
#             self.hidden_size, 
#             self.intermediate_size, 
#             bias=False,
#             device=device
#         )
        
#         self.down_proj = nn.Linear(
#             self.intermediate_size, 
#             self.hidden_size, 
#             bias=False,
#             device=device
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass with memory-efficient computation."""
#         return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# class AvaMLPWithExpertMoE(nn.Module):
#     """
#     Mixture of Experts MLP implementation for improved parameter efficiency.
#     Only used for larger model configurations (typically 13B+).
#     """
    
#     def __init__(
#         self,
#         config,
#         num_experts: int = 8,
#         num_experts_per_token: int = 2,
#         expert_capacity_factor: float = 1.0,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None
#     ):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.num_experts = num_experts
#         self.num_experts_per_token = num_experts_per_token
#         self.expert_capacity_factor = expert_capacity_factor
        
#         factory_kwargs = {'device': device, 'dtype': dtype}
        
#         # Expert gating network
#         self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, **factory_kwargs)
        
#         # Create expert networks (each is a SwiGLU MLP)
#         self.experts = nn.ModuleList([
#             AvaMLP(config, device=device, dtype=dtype) 
#             for _ in range(self.num_experts)
#         ])
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass with sparse MoE routing.
        
#         Args:
#             x: Input tensor of shape [batch_size, seq_length, hidden_size]
            
#         Returns:
#             Output tensor of shape [batch_size, seq_length, hidden_size]
#         """
#         batch_size, seq_length, hidden_size = x.shape
        
#         # Reshape input for routing
#         x_reshaped = x.view(-1, hidden_size)  # [batch_size*seq_length, hidden_size]
        
#         # Compute routing probabilities
#         router_logits = self.gate(x_reshaped)  # [batch_size*seq_length, num_experts]
        
#         # Get top-k experts per token
#         routing_weights, selected_experts = torch.topk(
#             router_logits, 
#             self.num_experts_per_token, 
#             dim=-1
#         )
#         routing_weights = F.softmax(routing_weights, dim=-1)
        
#         # Compute capacity per expert (how many tokens each expert can process)
#         tokens_per_expert = int(self.expert_capacity_factor * batch_size * seq_length * 
#                                self.num_experts_per_token / self.num_experts)
#         tokens_per_expert = max(1, tokens_per_expert)
        
#         # Create output tensor
#         final_output = torch.zeros_like(x_reshaped)
        
#         # Process tokens through experts
#         for expert_idx in range(self.num_experts):
#             # Find which tokens use this expert
#             expert_mask = (selected_experts == expert_idx)
            
#             # Get indices of tokens that use this expert
#             token_indices = torch.nonzero(expert_mask.any(dim=-1)).squeeze(-1)
            
#             # Skip if no tokens use this expert
#             if token_indices.shape[0] == 0:
#                 continue
                
#             # Apply capacity constraint
#             if token_indices.shape[0] > tokens_per_expert:
#                 # Select random subset of tokens if over capacity
#                 perm = torch.randperm(token_indices.shape[0], device=token_indices.device)
#                 token_indices = token_indices[perm[:tokens_per_expert]]
            
#             # Get expert weights for these tokens
#             expert_weights = routing_weights[token_indices, expert_mask[token_indices].nonzero()[:, 1]]
            
#             # Process these tokens with the expert
#             expert_output = self.experts[expert_idx](x_reshaped[token_indices])
            
#             # Combine outputs weighted by router probabilities
#             final_output.index_add_(
#                 0, 
#                 token_indices, 
#                 expert_output * expert_weights.unsqueeze(-1)
#             )
        
#         # Reshape back to original dimensions
#         return final_output.view(batch_size, seq_length, hidden_size)


# def create_mlp_block(
#     config, 
#     mlp_type: str = "default",
#     quantize: str = "none",
#     num_experts: int = 8,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None
# ) -> nn.Module:
#     """
#     Factory function to create the appropriate MLP implementation
#     based on configuration and hardware constraints.
    
#     Args:
#         config: AvaConfig instance
#         mlp_type: Type of MLP implementation ('default', 'quantized', 'moe')
#         quantize: Quantization level ('none', '4bit', '8bit')
#         num_experts: Number of experts for MoE models
#         device: Target device
#         dtype: Data type for weights
        
#     Returns:
#         Appropriate MLP implementation
#     """

#     if mlp_type == "quantized" or quantize != "none":
#         return AvaMLPQuantized(config, quantize=quantize, device=device)
    
#     elif mlp_type == "moe":
#         return AvaMLPWithExpertMoE(
#             config,
#             num_experts=num_experts,
#             device=device,
#             dtype=dtype
#         )
    
#     else:
#         return AvaMLP(
#             config,
#             device=device,
#             dtype=dtype,
#             use_memory_efficient_linear=(config.hidden_size >= 4096)
#         )