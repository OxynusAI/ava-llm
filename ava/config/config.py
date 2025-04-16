class AvaConfig:
    def __init__(
        self,
        vocab_size              = 32000,
        hidden_size             = 2048,
        intermediate_size       = 8192,
        num_hidden_layers       = 16,
        num_attention_heads     = 16,
        hidden_act              = 'silu',
        max_position_embeddings = 2048,
        initializer_range       = 0.02,
        rms_norm_eps            = 1e-5,
        use_cache               = True,
        pad_token_id            = 0,
        bos_token_id            = 1,
        eos_token_id            = 2,
        tie_word_embeddings     = False,
        rope_theta              = 10_000.0,
        attention_dropout       = 0.0,
        model_type              = 'ava',
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.model_type = model_type
        self.predefined_models = [
            'small',
            'medium',
            'base',
            '1b'
        ]
        
        if 'head_dim' in kwargs:
            self.head_dim = kwargs['head_dim']
        else:
            self.head_dim = hidden_size // num_attention_heads
            
        if 'kv_heads' in kwargs:
            self.kv_heads = kwargs['kv_heads']
        else:
            self.kv_heads = num_attention_heads
            
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


    def apply_for(self, model = 'base'):
        if model not in self.predefined_models:
            raise ValueError(f'Configuration for "{model}" is not defined.')
        
        if model == 'small':
            self.hidden_size = 768
            self.intermediate_size = 3072
            self.num_hidden_layers = 6
            self.num_attention_heads = 12
            self.max_position_embeddings = 512
            self.head_dim = 64
            self.kv_heads = 4

        elif model == 'medium':
            self.hidden_size = 1024
            self.intermediate_size = 4096
            self.num_hidden_layers = 12
            self.num_attention_heads = 16
            self.max_position_embeddings = 1024
            self.head_dim = 64
            self.kv_heads = 8

        elif model == 'base':
            self.hidden_size = 1536
            self.intermediate_size = 6144
            self.num_hidden_layers = 12
            self.num_attention_heads = 16
            self.max_position_embeddings = 2048
            self.head_dim = 96
            self.kv_heads = 8

        elif model == '1b':
            self.hidden_size = 2048
            self.intermediate_size = 8192
            self.num_hidden_layers = 16
            self.num_attention_heads = 16
            self.max_position_embeddings = 2048
            self.head_dim = 128
            self.kv_heads = 8

        return self