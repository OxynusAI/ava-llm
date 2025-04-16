import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AvaAttention
from .normalization import AvaRMSNorm
from .embeddings import AvaRotaryEmbedding
from .mlp import AvaMLP

class AvaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.self_attn = AvaAttention(config)
        self.mlp = AvaMLP(config)
        self.input_layernorm = AvaRMSNorm(
            config.hidden_size, 
            epsilon = config.rms_norm_eps
        )
        
        self.post_attention_layernorm = AvaRMSNorm(
            config.hidden_size, 
            epsilon = config.rms_norm_eps
        )
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        rotary_emb=None,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention block
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            rotary_emb=rotary_emb,
        )
        
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (attn_outputs[1],)  # past_key_value
        
        if output_attentions:
            outputs += (attn_outputs[2],)  # attention_probs
            
        return outputs

# Ava Model
class AvaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([AvaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.rotary_emb = AvaRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            position_ids = position_ids[:, past_length:]
            
        # Prepare attention mask
        if attention_mask is not None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), -float("inf"), device=attention_mask.device),
                diagonal=1,
            )
            # Extend attention mask to causal mask
            expanded_attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(torch.float32).min
            expanded_attn_mask = expanded_attn_mask + causal_mask.unsqueeze(0)
        else:
            # Causal mask
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), -float("inf"), device=input_ids.device),
                diagonal=1,
            )
            expanded_attn_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        hidden_states = inputs_embeds
        
        # Prepare for RoPE
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_past_key_values = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=expanded_attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                rotary_emb=self.rotary_emb,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_past_key_values += (layer_outputs[1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                
        # Final layernorm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_past_key_values,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
        }

# LM Head
class AvaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AvaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Weight tying (optional)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
        
        hidden_states = output['last_hidden_state']
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': output.get('past_key_values', None),
            'hidden_states': output.get('hidden_states', None),
            'attentions': output.get('attentions', None),
        }
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        
        # If past_key_values are used, only take the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        # Create position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids only for non-padded tokens
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
        input_ids,
        attention_mask=None,
        max_length=None,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
    ):
        # Set default values
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        max_length = max_length if max_length is not None else self.config.max_position_embeddings
        
        batch_size = input_ids.shape[0]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        input_ids_seq_length = input_ids.shape[-1]
        
        # Store generated tokens
        generated_tokens = input_ids.clone()
        
        # Initialize past key values
        past_key_values = None
        
        # Keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        for current_length in range(input_ids_seq_length, max_length):
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                generated_tokens,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
            
            # Forward pass
            outputs = self.forward(**model_inputs)
            
            # Get logits for the next token
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Update past key values
            past_key_values = outputs['past_key_values']
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_tokens[i]:
                        # Where the token is repeated, reduce its probability
                        if previous_token in [pad_token_id, eos_token_id]:
                            continue
                        next_token_logits[i, previous_token] /= repetition_penalty
                        
            # Apply Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Apply Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
            # Set tokens to eos_token_id if sequences are finished
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + eos_token_id * (1 - unfinished_sequences)
                
            # Update generated tokens
            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)
            
            # Mark finished sequences
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                
            # Stop when all sequences are finished
            if unfinished_sequences.max() == 0:
                break
                
        return generated_tokens
