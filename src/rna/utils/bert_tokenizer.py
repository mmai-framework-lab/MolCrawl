"""
RNA用のBERT互換トークナイザーラッパー
TranscriptomeTokenizerをBERT学習と互換性のある形式でラップ
"""

from rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer
from typing import Dict, Any, Union, List, Optional
import torch


class BertRnaTokenizer:
    """
    RNA TranscriptomeTokenizer wrapper for BERT compatibility
    
    This class wraps the TranscriptomeTokenizer to make it compatible
    with BERT training and testing pipelines
    """
    
    # Override model input names to use standard BERT format
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(self, **kwargs):
        self.tokenizer = TranscriptomeTokenizer(**kwargs)
        
        # BERT compatibility attributes
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        
        # Create a simple vocab mapping for testing purposes
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }
        
        # Add gene tokens from the underlying tokenizer
        if hasattr(self.tokenizer, 'gene_token_dict'):
            for gene_id, token_id in self.tokenizer.gene_token_dict.items():
                # Offset by 5 to account for special tokens
                adjusted_token_id = token_id + 5 if isinstance(token_id, int) else token_id
                self.vocab[gene_id] = adjusted_token_id
    
    def get_vocab(self):
        """Get vocabulary dictionary"""
        return self.vocab
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (for RNA, this is mostly for compatibility)
        """
        # For RNA data, we typically work with gene expression vectors
        # This is a simplified tokenization for testing purposes
        if isinstance(text, str):
            # Simple word-level tokenization for testing
            tokens = text.split()
            return tokens
        return []
    
    def encode(self, text: Union[str, List[str]], 
               add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               padding: bool = False,
               truncation: bool = False,
               return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs
        """
        if isinstance(text, str):
            tokens = self.tokenize(text)
        else:
            tokens = text
        
        # Convert tokens to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.cls_token_id)
        
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.sep_token_id)
        
        # Apply truncation
        if max_length and truncation:
            if len(token_ids) > max_length:
                if add_special_tokens:
                    # Keep CLS token and ensure SEP token at the end
                    token_ids = token_ids[:max_length-1] + [self.sep_token_id]
                else:
                    token_ids = token_ids[:max_length]
        
        # Apply padding
        if max_length and padding:
            while len(token_ids) < max_length:
                token_ids.append(self.pad_token_id)
        
        if return_tensors == "pt":
            return torch.tensor(token_ids)
        
        return token_ids
    
    def __call__(self, 
                 text: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False,
                 max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Tokenize and encode text
        """
        if isinstance(text, list):
            # Batch processing
            input_ids = []
            attention_masks = []
            
            for single_text in text:
                encoded = self.encode(
                    single_text,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=None
                )
                input_ids.append(encoded)
                
                # Create attention mask
                attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in encoded]
                attention_masks.append(attention_mask)
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_masks
            }
            
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor(result["input_ids"])
                result["attention_mask"] = torch.tensor(result["attention_mask"])
        
        else:
            # Single text processing
            input_ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None
            )
            
            # Create attention mask
            attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor([result["input_ids"]])
                result["attention_mask"] = torch.tensor([result["attention_mask"]])
        
        return result
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        
        # Create reverse mapping
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            token = id_to_token.get(token_id, self.unk_token)
            
            if skip_special_tokens and token in [self.pad_token, self.cls_token, self.sep_token]:
                continue
                
            tokens.append(token)
        
        return " ".join(tokens)
    
    def pad(self, encoded_inputs, **kwargs):
        """
        Pad encoded inputs (for compatibility)
        """
        return encoded_inputs


def create_bert_rna_tokenizer(**kwargs) -> BertRnaTokenizer:
    """
    Create a BERT-compatible RNA tokenizer
    
    Returns:
        BertRnaTokenizer instance
    """
    return BertRnaTokenizer(**kwargs)
