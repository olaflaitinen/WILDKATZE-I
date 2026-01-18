from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
import sentencepiece as spm
import os

class WildkatzeTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: str,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<unk>",
        **kwargs,
    ):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.vocab_file = vocab_file
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.DecodePieces(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Union[str, tuple]:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model")
        
        # In a real implementation we would copy the .model file
        # shutil.copyfile(self.vocab_file, vocab_file)
        return (vocab_file,)
