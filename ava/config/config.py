import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Literal


@dataclass
class AvaConfig:
    '''
    Configuration class for Ava language models.
    
    This class provides configuration parameters for different sized models,
    with improved memory efficiency, type annotations, and flexible scaling options.
    '''

    vocab_size:              int = 32000
    hidden_size:             int = 2048
    intermediate_size:       int = 8192
    num_hidden_layers:       int = 16
    num_attention_heads:     int = 16
    hidden_act:              str = 'silu'
    max_position_embeddings: int = 2048
    initializer_range:       float = 0.02
    rms_norm_eps:            float = 1e-5
    use_cache:               bool = True
    pad_token_id:            int = 0
    bos_token_id:            int = 1
    eos_token_id:            int = 2
    tie_word_embeddings:     bool = False
    rope_theta:              float = 10_000.0
    attention_dropout:       float = 0.0
    model_type:              str = 'ava'
    head_dim:                Optional[int] = None
    kv_heads:                Optional[int] = None
    
    predefined_models: List[str] = field(default_factory=lambda: [
        '100m', '500m', '1b', 
        '3b', '7b', '13b', 
        '30b', '65b', '100b'
    ])
    
    model_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'tiny': [
            '100m', '500m'
        ],
        
        'small': [
            '1b', '3b'
        ],
        
        'medium': [
            '7b', '13b'
        ],
        
        'large': [
            '30b', '65b'
        ],

        'massive': ['100b']
    })
    
    def __post_init__(self):
        '''Initialize derived parameters after object creation.'''

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
            
        if self.kv_heads is None:
            self.kv_heads = self.num_attention_heads
            
            if self.hidden_size >= 4096:
                self.kv_heads = max(8, self.num_attention_heads // 4)
    
    def to_dict(self) -> Dict:
        '''Convert configuration to dictionary format.'''

        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AvaConfig':
        '''Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            An AvaConfig instance with the specified parameters
        '''

        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)
    
    def scale_model(self, 
                    scale_factor: float = 1.0, 
                    base_model: str = '7b',
                    width_scale: Optional[float] = None,
                    depth_scale: Optional[float] = None) -> 'AvaConfig':
        '''
        Scale a model configuration by a custom factor.
        
        Args:
            scale_factor: Overall scaling factor for model size
            base_model: The base model to scale from
            width_scale: Optional specific scaling factor for width (hidden size)
            depth_scale: Optional specific scaling factor for depth (layers)
            
        Returns:
            A new scaled AvaConfig instance
        '''

        config = self.apply_for(base_model)
        
        if width_scale is None:
            width_scale = math.sqrt(scale_factor)
        
        if depth_scale is None:
            depth_scale = math.sqrt(scale_factor)
            
        scaled_config = AvaConfig(**self.to_dict())
        scaled_config.hidden_size = int(config.hidden_size * width_scale)
        scaled_config.intermediate_size = int(config.intermediate_size * width_scale)
        
        orig_head_dim = config.hidden_size // config.num_attention_heads
        scaled_config.num_attention_heads = max(1, scaled_config.hidden_size // orig_head_dim)
        scaled_config.hidden_size = scaled_config.num_attention_heads * orig_head_dim
        
        scaled_config.num_hidden_layers = max(1, int(config.num_hidden_layers * depth_scale))
        
        if scaled_config.num_attention_heads >= 32:
            scaled_config.kv_heads = min(scaled_config.num_attention_heads // 4, 32)
            scaled_config.kv_heads = max(8, scaled_config.kv_heads)  # At least 8 KV heads
        else:
            scaled_config.kv_heads = max(1, scaled_config.num_attention_heads // 2)
            
        scaled_config.head_dim = scaled_config.hidden_size // scaled_config.num_attention_heads
        
        base_ctx = config.max_position_embeddings
        ctx_scale = math.log(1 + scale_factor) / math.log(2)  # Logarithmic scaling
        scaled_config.max_position_embeddings = min(32768, int(base_ctx * (1 + ctx_scale)))
        
        return scaled_config
    
        
    def apply_for(self, model: str = '7b') -> 'AvaConfig':
        '''
        Configure for a predefined model size.
        
        Args:
            model: Model size identifier (e.g., '7b', '13b')
            
        Returns:
            Self with updated configuration
            
        Raises:
            ValueError: If the model size is not predefined
        '''

        if model not in self.predefined_models:
            raise ValueError(f'Configuration for \'{model}\' is not defined. '
                           f'Available models: {", ".join(self.predefined_models)}')

        # Tiny models (Edge devices, IoT, offline agents, chatbots)
        if model == '100m':
            self.hidden_size = 768
            self.intermediate_size = 3072
            self.num_hidden_layers = 6
            self.num_attention_heads = 12
            self.max_position_embeddings = 2048
            self.head_dim = 64
            self.kv_heads = 4

        elif model == '500m':
            self.hidden_size = 1024
            self.intermediate_size = 4096
            self.num_hidden_layers = 8
            self.num_attention_heads = 16
            self.max_position_embeddings = 2048
            self.head_dim = 64
            self.kv_heads = 4

        elif model == '1b':
            self.hidden_size = 1280
            self.intermediate_size = 5120
            self.num_hidden_layers = 12
            self.num_attention_heads = 16
            self.max_position_embeddings = 4096
            self.head_dim = 80
            self.kv_heads = 8

        elif model == '3b':
            self.hidden_size = 1600
            self.intermediate_size = 6400
            self.num_hidden_layers = 24
            self.num_attention_heads = 16
            self.max_position_embeddings = 4096
            self.head_dim = 100
            self.kv_heads = 8

        # Medium models (Coding, reasoning, multi-turn chat, translation)
        elif model == '7b':
            self.hidden_size = 4096
            self.intermediate_size = 11008
            self.num_hidden_layers = 32
            self.num_attention_heads = 32
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        elif model == '13b':
            self.hidden_size = 5120
            self.intermediate_size = 13824
            self.num_hidden_layers = 40
            self.num_attention_heads = 40
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        # Large models (Research, enterprise-level applications)
        elif model == '30b':
            self.hidden_size = 6656
            self.intermediate_size = 17920
            self.num_hidden_layers = 60
            self.num_attention_heads = 52
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        elif model == '65b':
            self.hidden_size = 8192
            self.intermediate_size = 22016
            self.num_hidden_layers = 80
            self.num_attention_heads = 64
            self.max_position_embeddings = 8192
            self.head_dim = 128
            self.kv_heads = 8

        # Massive models (AGI research, cutting-edge LLMs)
        elif model == '100b':
            self.hidden_size = 12288
            self.intermediate_size = 33024
            self.num_hidden_layers = 96
            self.num_attention_heads = 96
            self.max_position_embeddings = 16384
            self.head_dim = 128
            self.kv_heads = 8

        return self
    
    def get_params_count(self, millions: bool = True) -> Union[int, float]:
        '''
        Estimate the number of parameters in the model.
        
        Args:
            millions: Whether to return the count in millions
            
        Returns:
            Parameter count as either raw count or millions
        '''
        
        embedding_params = self.vocab_size * self.hidden_size
        
        per_layer_params = (
            (self.hidden_size * self.head_dim * (self.num_attention_heads + 2 * self.kv_heads)) + 
            (self.num_attention_heads * self.head_dim * self.hidden_size) +
            (self.hidden_size * self.intermediate_size) + 
            (self.intermediate_size * self.hidden_size) + 
            (2 * self.hidden_size)
        )
        
        output_params = self.hidden_size + (self.hidden_size * self.vocab_size)
        total_params = embedding_params + (per_layer_params * self.num_hidden_layers) + output_params
        
        return total_params / 1_000_000 if millions else total_params
    
    def get_category(self) -> str:
        '''
        Determine the category of the current model configuration.
        
        Returns:
            Category name: 'tiny', 'small', 'medium', 'large', or 'massive'
        '''
        param_count = self.get_params_count(millions=True)
        
        if param_count < 1_000:
            return 'tiny'
            
        elif param_count < 5_000:
            return 'small'
            
        elif param_count < 20_000:
            return 'medium'
            
        elif param_count < 80_000:
            return 'large'
            
        else: 
            return 'massive'
    
    def optimize_for_inference(self, device_memory_gb: float = 16.0) -> 'AvaConfig':
        '''
        Optimize configuration for inference on specific hardware.
        
        Args:
            device_memory_gb: Available device memory in GB
            
        Returns:
            Optimized configuration for efficient inference
        '''

        config = AvaConfig(**self.to_dict())
        params_count = self.get_params_count(millions=False)
        approx_model_size_gb = (params_count * 2) / (1024 ** 3) 

        if approx_model_size_gb > device_memory_gb * 0.8: 
            config.kv_heads = max(1, config.num_attention_heads // 8)
            
            if config.max_position_embeddings > 4096:
                config.max_position_embeddings = 4096
        
        config.use_cache = True
        
        return config

def create_custom_model(params_target: Union[int, float], 
                        base_model: str = '7b',
                        width_depth_ratio: float = 1.0) -> AvaConfig:
    '''
    Create a custom-sized model targeting a specific parameter count.
    
    Args:
        params_target: Target parameter count in billions
        base_model: Base model to scale from
        width_depth_ratio: Ratio of width scaling to depth scaling
                          (>1 favors wider models, <1 favors deeper models)
    
    Returns:
        AvaConfig configured for the target parameter count
    '''

    config = AvaConfig()
    base_config = config.apply_for(base_model)
    base_params = base_config.get_params_count()
    
    scale_factor = params_target / base_params
    
    total_scale = math.sqrt(scale_factor)
    width_scale = total_scale * math.sqrt(width_depth_ratio)
    depth_scale = total_scale / math.sqrt(width_depth_ratio)
    
    return config.scale_model(
        scale_factor = scale_factor,
        base_model   = base_model,
        width_scale  = width_scale,
        depth_scale  = depth_scale
    )
