import torch
from transformers import AutoTokenizer

class S3CMNTokenizer:
    """Tokenizer for S³CMN model."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = self.tokenizer.encode(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        return tokens.squeeze(0)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else self.tokenizer.sep_token_id

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else self.tokenizer.sep_token_id
