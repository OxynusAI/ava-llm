import json
import numpy as np

from .dataset import DialogueDataset

def prepare_data_from_json(
        file_path, 
        tokenizer, 
        train_ratio = 0.9, 
        max_length = 512
    ):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    np.random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = DialogueDataset(train_data, tokenizer, max_length)
    val_dataset = DialogueDataset(val_data, tokenizer, max_length)
    
    return train_dataset, val_dataset
