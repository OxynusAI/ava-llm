class SimpleTokenizer:
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
        self.pad_token_id = base_tokenizer.pad_token_id
        self.bos_token_id = base_tokenizer.bos_token_id
        self.eos_token_id = base_tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def batch_decode(self, sequences, **kwargs):
        return self.tokenizer.batch_decode(sequences, **kwargs)
