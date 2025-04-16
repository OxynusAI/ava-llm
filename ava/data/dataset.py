from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    """Dataset preparation"""
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.labels = []
        
        for dialogue in data:
            full_text = ''

            for message in dialogue:
                content = message['content']

                if message['role'] == 'user':
                    full_text += f'User: {content}\n'

                elif message['role'] == 'assistant':
                    full_text += f'Ava: {content}\n'
                    
            encodings = self.tokenizer(
                full_text,
                max_length     = self.max_length,
                padding        = 'max_length',
                truncation     = True,
                return_tensors = 'pt'
            )
            
            self.encodings.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze()
            })
            
            self.labels.append(encodings['input_ids'].squeeze())
            
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'labels': self.labels[idx]
        }
