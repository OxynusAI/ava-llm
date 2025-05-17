from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    Dataset for pretraining a language model. It tokenizes the input texts and prepares them for training.
    """

    def __init__(self, texts, tokenizer, max_length=256):
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
                f'{tokenizer.bos_token}{text.strip()}{tokenizer.eos_token}',
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
