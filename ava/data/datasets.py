import torch

from torch.utils.data import Dataset

class AvaDataset(Dataset):
    def __init__(
        self, 
        data, 
        tokenizer,
        max_length=256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for conversation in data:
            if len(conversation) < 2 or 'content' not in conversation[0] or 'content' not in conversation[1]:
                continue
                
            user = conversation[0]['content']
            assistant = conversation[1]['content']
            
            full_text = f'User: {user}\nAssistant: {assistant}'
            
            encoding = self.tokenizer(
                full_text,
                truncation     = True,
                max_length     = self.max_length,
                padding        = 'max_length',
                return_tensors = 'pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            if torch.max(input_ids).item() >= len(tokenizer):
                print(f'⚠ Warning: Found token ID {torch.max(input_ids).item()} which is >= vocabulary size {len(tokenizer)}')
                continue
            
            labels = input_ids.clone()
            
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }