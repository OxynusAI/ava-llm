import torch

from torch import nn
from typing import Any, List, Sequence, Tuple, Union

from core import AvaConfig, AvaTokenizer
from core.layers import AvaDecoderLayer, RMSNorm, Embedding
from core.utils import AvaSampler, precompute_freqs_cis

class Ava(nn.Module):
    def __init__(self, config: AvaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList()
        
        for _ in range(config.num_hidden_layers):
            self.layers.append(AvaDecoderLayer(config))
        
        self.norm = RMSNorm(
            config.hidden_size, 
            eps = config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            
            hidden_states = layer(
                hidden_states    = hidden_states,
                freqs_cis        = freqs_cis,
                kv_write_indices = kv_write_indices,
                kv_cache         = kv_caches[i],
                mask             = mask,
            )
            
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    
class AvaForCausalLM(nn.Module):
    def __init__(
        self,
        config: AvaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = AvaTokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = Ava(config)
        self.sampler = AvaSampler(vocab_size)

        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(
            head_dim,
            max_seq_len * 2,
            theta = rope_theta
        )
        
        self.register_buffer('freqs_cis', freqs_cis)

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions
        hidden_states = self.embedder(input_token_ids)
        hidden_states = hidden_states * (self.config.hidden_size ** 0.5)

        hidden_states = self.model(
            hidden_states    = hidden_states,
            freqs_cis        = freqs_cis,
            kv_write_indices = kv_write_indices,
            kv_caches        = kv_caches,
            mask             = mask,
        )
        
        embedder_weight = self.embedder.weight
        
        if self.config.quant:
            embedder_weight = (embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
            
        next_tokens = self.sampler(
            embedding        = embedder_weight,
            hidden_states    = hidden_states,
            output_positions = output_positions,
            temperatures     = temperatures,
            top_ps           = top_ps,
            top_ks           = top_ks,
        )
        
        return next_tokens

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> Union[str, Sequence[str]]:
        is_str_prompt = isinstance(prompts, str)
        
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        
        assert max_seq_len <= self.config.max_position_embeddings

        kv_caches = []
        
        for _ in range(self.config.num_hidden_layers):
            size = (
                batch_size, max_seq_len, 
                self.config.num_key_value_heads,
                self.config.head_dim
            )
            
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        token_ids_tensor = torch.full(
            (
                batch_size, 
                max_seq_len
            ),
            
            self.tokenizer.pad_id, 
            dtype = torch.int64
        )
        
        input_token_ids_tensor = torch.full(
            (
                batch_size, 
                min_prompt_len
            ),
            
            self.tokenizer.pad_id,
            dtype = torch.int64
        )
        
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
        
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        
        input_positions_tensor = torch.arange(
            0, 
            min_prompt_len,
            dtype=torch.int64
        ).to(device)
        
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        for i in range(max_seq_len - min_prompt_len):
            next_token_ids = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1

        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
        self.load_state_dict(
            torch.load(
                model_path, mmap=True, weights_only=True,
            )['model_state_dict'],
            strict=False,
        )
