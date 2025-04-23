import torch

from torch.utils.data import Dataset

class AvaDataset(Dataset):
    """
    Dataset for training a language model with conversational dataset like ChatLM. It tokenizes the input texts and prepares them for training.
    """

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
    

class PretrainDataset(Dataset):
    """
    Dataset for pretraining a language model. It tokenizes the input texts and prepares them for training.
    """

    def __init__(
            self, 
            texts, 
            tokenizer, 
            max_length = 256
        ):
        """
        PretrainDataset is a PyTorch Dataset class for pretraining a language model.
        It takes a list of texts and tokenizes them using the provided tokenizer.

        Args:
            texts (list): List of input texts to be tokenized.
            tokenizer: Tokenizer to be used for encoding the texts.
            max_length (int): Maximum length of the tokenized sequences.
        """

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text in texts:
            encoding = tokenizer(
                text, 
                truncation     = True, 
                padding        = 'max_length', 
                max_length     = self.max_length, 
                return_tensors = 'pt'
            )

            self.samples.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': encoding['input_ids'].squeeze(0)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
