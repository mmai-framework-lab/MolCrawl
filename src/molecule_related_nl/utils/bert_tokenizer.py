"""
Molecule Natural Language用のBERT互換トークナイザーラッパー
MoleculeNatLangTokenizerをBERT学習と互換性のある形式でラップ
"""

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer
from typing import Dict, Any, Union, List, Optional
import torch


class BertMoleculeNlTokenizer:
    """
    MoleculeNatLangTokenizer wrapper for BERT compatibility

    This class wraps the MoleculeNatLangTokenizer to make it compatible
    with BERT training and testing pipelines
    """

    # Override model input names to use standard BERT format
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        self.tokenizer = MoleculeNatLangTokenizer(**kwargs)

        # BERT compatibility attributes
        self.pad_token = self.tokenizer.tokenizer.pad_token
        self.unk_token = self.tokenizer.tokenizer.unk_token
        self.cls_token = getattr(self.tokenizer.tokenizer, "cls_token", "[CLS]")
        self.sep_token = getattr(self.tokenizer.tokenizer, "sep_token", "[SEP]")
        self.mask_token = getattr(self.tokenizer.tokenizer, "mask_token", "[MASK]")

        self.pad_token_id = self.tokenizer.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.tokenizer.unk_token_id
        self.cls_token_id = getattr(self.tokenizer.tokenizer, "cls_token_id", 101)
        self.sep_token_id = getattr(self.tokenizer.tokenizer, "sep_token_id", 102)
        self.mask_token_id = getattr(self.tokenizer.tokenizer, "mask_token_id", 103)

    def get_vocab(self):
        """Get vocabulary dictionary"""
        return self.tokenizer.tokenizer.get_vocab()

    def __len__(self):
        """Return vocabulary size"""
        return len(self.tokenizer.tokenizer)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the underlying tokenizer
        """
        return self.tokenizer.tokenizer.tokenize(text)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs
        """
        return self.tokenizer.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Tokenize and encode text using the underlying tokenizer
        """
        return self.tokenizer.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text
        """
        return self.tokenizer.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens to string
        """
        return self.tokenizer.tokenizer.convert_tokens_to_string(tokens)

    def pad(self, encoded_inputs, **kwargs):
        """
        Pad encoded inputs
        """
        return self.tokenizer.tokenizer.pad(encoded_inputs, **kwargs)


def create_bert_molecule_nl_tokenizer(**kwargs) -> BertMoleculeNlTokenizer:
    """
    Create a BERT-compatible molecule natural language tokenizer

    Returns:
        BertMoleculeNlTokenizer instance
    """
    return BertMoleculeNlTokenizer(**kwargs)
