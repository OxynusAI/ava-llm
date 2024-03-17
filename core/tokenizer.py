import os

from typing import List, Optional
from sentencepiece import SentencePieceProcessor


class AvaTokenizer:
    def __init__(self, model_path: Optional[str]):
        assert os.path.isfile(model_path), model_path
        self.sp_processor = SentencePieceProcessor(model_file=model_path)

        self.num_words: int = self.sp_processor.vocab_size()
        self.beginning_of_sentence_id: int = self.sp_processor.bos_id()
        self.end_of_sentence_id: int = self.sp_processor.eos_id()
        self.padding_id: int = self.sp_processor.pad_id()
        
        assert self.sp_processor.vocab_size() == self.sp_processor.get_piece_size()

    def tokenize(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        assert isinstance(text, str)
        
        tokenized_text = self.sp_processor.encode(text)
        
        if add_bos:
            tokenized_text = [self.beginning_of_sentence_id] + tokenized_text
        
        if add_eos:
            tokenized_text = tokenized_text + [self.end_of_sentence_id]
        
        return tokenized_text

    def detokenize(self, tokenized_text: List[int]) -> str:
        return self.sp_processor.decode(tokenized_text)
