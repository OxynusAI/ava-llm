import os
import torch
import json
import traceback
import numpy as np

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from ava import AvaConfig, AvaForCausalLM
from ava.data.datasets import AvaDataset
from ava.training.trainer import train_model
from ava.utils import collate_fn

config = AvaConfig().apply_for('100m')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config.vocab_size = len(tokenizer)
config.pad_token_id = tokenizer.pad_token_id
config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
config.eos_token_id = tokenizer.eos_token_id

print(f'Tokenizer vocabulary size: {len(tokenizer)}')
print(f'Config vocabulary size: {config.vocab_size}')

with open('../data/oasst1_english_conversations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

valid_data = []
for conv in data:
    if isinstance(conv, list) and len(conv) > 0:
        valid_data.append(conv[0])

print(f'Found {len(valid_data)}/{len(data)} valid conversations')

np.random.shuffle(valid_data)
split_idx = int(len(valid_data) * 0.9)
train_data = valid_data[:split_idx]
val_data = valid_data[split_idx:]


max_seq_length = 256
train_dataset = AvaDataset(train_data, tokenizer, max_length=max_seq_length)
val_dataset = AvaDataset(val_data, tokenizer, max_length=max_seq_length)

print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError('Dataset is empty after processing. Check data format and filtering.')

batch_size = 2
train_loader = DataLoader(
    train_dataset, 
    batch_size = batch_size, 
    shuffle    = True,
    collate_fn = collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size = batch_size,
    collate_fn = collate_fn
)

sample_batch = next(iter(train_loader))

print(f'Sample batch shapes:')
print(f'input_ids: {sample_batch["input_ids"].shape}')
print(f'attention_mask: {sample_batch["attention_mask"].shape}')
print(f'labels: {sample_batch["labels"].shape}')

max_token_id = torch.max(sample_batch['input_ids']).item()
print(f'Maximum token ID in batch: {max_token_id}')
print(f'Tokenizer vocabulary size: {len(tokenizer)}')

if max_token_id >= len(tokenizer):
    raise ValueError(f'Maximum token ID {max_token_id} is out of range for vocabulary size {len(tokenizer)}')

model = AvaForCausalLM(config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr = 5e-5, 
    weight_decay = 0.01
)  

try:
    train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        num_epochs   = 2,
        device       = device
    )
    
    torch.save(model.state_dict(), 'ava_model_trained.pt')
    
except Exception as e:
    print(f'❌ Training error: {e}')
    traceback.print_exc()

except KeyboardInterrupt:
    print('🙄 As you wish, Sir!')

# input_text = 'User: What is AI?\nAssistant:'
# input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# try:
#     output = model.generate(
#         input_ids,
#         max_length=100,
#         temperature=0.7,
#         top_p=0.9
#     )
    
#     print(tokenizer.decode(output[0]))
# except Exception as e:
#     print(f'❌ Generation error: {e}')
#     traceback.print_exc()


