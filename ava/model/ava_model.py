from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import AvaRotaryEmbedding
from .normalization import AvaRMSNorm
from .mlp import AvaMLP
from .attention import AvaAttention

class AvaDecoderLayer(nn.Module):
    """
    Enhanced decoder layer with performance optimizations and improved architecture
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.input_layernorm = AvaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )
        self.post_attention_layernorm = AvaRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps
        )

        self.self_attn = AvaAttention(config)
        self.mlp = AvaMLP(config)

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

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            rotary_emb=rotary_emb,
        )
        

        hidden_states = residual + attn_outputs[0]
        

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (attn_outputs[1],)
        
        if output_attentions:
            outputs += (attn_outputs[-1],)
            
        return outputs


class AvaModel(nn.Module):
    """
    Enhanced base model implementation with modern architecture features
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
        
        self.layers = nn.ModuleList(
            [AvaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        rope_scaling_factor = getattr(config, "rope_scaling_factor", 1.0)

        self.rotary_emb = AvaRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling_factor
        )
        

        self.apply(self._init_weights)
        self.gradient_checkpointing = False
        
    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            
    def _setup_gradient_checkpointing(self, enable=False):
        """Setup gradient checkpointing for memory efficiency during training"""

        self.gradient_checkpointing = enable
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape

        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape

        else:
            raise ValueError("You must provide either input_ids or inputs_embeds")
            

        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
                
            position_ids = torch.arange(
                past_length, seq_length + past_length, 
                dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)
        

        if attention_mask is not None:

            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0
                
            if len(attention_mask.shape) == 2: 
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                causal_mask = torch.triu(
                    torch.full((seq_length, seq_length), float("-inf"), device=attention_mask.device), 
                    diagonal=1
                )

                if past_length > 0:
                    causal_mask = torch.cat(
                        [
                            torch.zeros((seq_length, past_length), device=attention_mask.device),
                            causal_mask,
                        ],
                        dim=-1,
                    )
                    

                mask_value = torch.tensor(float("-inf"), dtype=torch.float32, device=attention_mask.device)
                attention_mask = (1.0 - extended_mask) * mask_value + extended_mask * causal_mask.unsqueeze(0)
                

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[idx] if past_key_values is not None else None
                
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    self.rotary_emb,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    rotary_emb=self.rotary_emb,
                )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[-1],)
                

        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class AvaForCausalLM(nn.Module):
    """
    Enhanced causal language model with improved generation capabilities
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AvaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens
        
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = output["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": output.get("past_key_values", None),
            "hidden_states": output.get("hidden_states", None),
            "attentions": output.get("attentions", None),
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """
        Prepare inputs for efficient generation
        """

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)


        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        streamer: Optional[Any] = None,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
        typical_p: Optional[float] = None,
    ):
        """
        Enhanced generate method with improved sampling strategies and streaming
        """

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        max_length = max_length if max_length is not None else self.config.max_position_embeddings

        batch_size = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        input_ids_seq_length = input_ids.shape[-1]
        generated_tokens = input_ids.clone()
        
        cached_position_ids = torch.arange(
            input_ids_seq_length, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        past_key_values = None
        self.eval()
        
        banned_ngram_tokens = [{} for _ in range(batch_size)] if no_repeat_ngram_size > 0 else None
        
        for current_length in range(input_ids_seq_length, max_length):
            if past_key_values is not None:
                inputs = generated_tokens[:, -1].unsqueeze(-1)

            else:
                inputs = generated_tokens
                
            if past_key_values is not None:
                position_ids = cached_position_ids[:, -1].unsqueeze(-1) + 1
                cached_position_ids = torch.cat([cached_position_ids, position_ids], dim=-1)

            else:
                position_ids = cached_position_ids
                

            if attention_mask is not None and past_key_values is not None:
                attention_mask = torch.cat(
                    [attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1
                )
                
            model_inputs = self.prepare_inputs_for_generation(
                inputs,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )
            
            outputs = self(
                **model_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            
            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"] if use_cache else None
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            if repetition_penalty != 1.0:
                token_ids = generated_tokens.tolist()

                for batch_idx, batch_token_ids in enumerate(token_ids):
                    for token_idx in set(batch_token_ids):
                        next_token_logits[batch_idx, token_idx] /= repetition_penalty
            

            if no_repeat_ngram_size > 0:
                for batch_idx in range(batch_size):
                    banned_tokens = self._get_banned_ngram_tokens(
                        generated_tokens[batch_idx].tolist(),
                        batch_idx,
                        no_repeat_ngram_size,
                        banned_ngram_tokens,
                    )

                    for banned_token in banned_tokens:
                        next_token_logits[batch_idx, banned_token] = float("-inf")
            

            if do_sample:
                if typical_p is not None:
                    probs = F.softmax(next_token_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    typical_scores = torch.abs(log_probs - entropy)
                    sorted_indices = torch.argsort(typical_scores, dim=-1)
                    cumulative_probs = torch.cumsum(
                        probs.gather(-1, sorted_indices), dim=-1
                    )
                    
                    typical_mask = cumulative_probs <= typical_p
                    typical_mask = torch.cat(
                        [typical_mask.new_ones(typical_mask.shape[:-1] + (1,)), typical_mask[:, :-1]], dim=-1
                    )

                    sorted_indices_to_remove = ~typical_mask
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    below_top_k_mask = torch.ones_like(next_token_logits).bool()
                    
                    for batch_idx in range(batch_size):
                        below_top_k_mask[batch_idx, top_k_indices[batch_idx]] = False
                    
                    next_token_logits.masked_fill_(below_top_k_mask, float("-inf"))
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = float("-inf")
                

                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            

            if eos_token_id is not None:
                tokens_to_add = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)

            else:
                tokens_to_add = next_tokens

            generated_tokens = torch.cat([generated_tokens, tokens_to_add.unsqueeze(-1)], dim=-1)

            if streamer is not None:
                streamer.put(tokens_to_add.unsqueeze(-1))
                
            if unfinished_sequences.max() == 0 or generated_tokens.shape[-1] >= max_length:
                break

        if streamer is not None:
            streamer.end()
            
        return generated_tokens

    def _get_banned_ngram_tokens(
        self, 
        prev_tokens, 
        batch_idx, 
        no_repeat_ngram_size, 
        banned_ngram_tokens
    ):
        """Helper function to identify banned n-grams for no-repeat n-gram suppression"""

        if len(prev_tokens) < no_repeat_ngram_size:
            return []
            
        ngram = tuple(prev_tokens[-(no_repeat_ngram_size-1):])
        banned_tokens = banned_ngram_tokens[batch_idx].get(ngram, [])

        if banned_tokens:
            return banned_tokens
            
        for i in range(len(prev_tokens) - no_repeat_ngram_size + 1):
            if tuple(prev_tokens[i:i+no_repeat_ngram_size-1]) == ngram:
                banned_tokens.append(prev_tokens[i+no_repeat_ngram_size-1])
                
        banned_ngram_tokens[batch_idx][ngram] = banned_tokens
        return banned_tokens
