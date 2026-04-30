"""
BERT-compatible tokenizer wrapper for Protein Sequence
Wrap the ESM tokenizer in a format compatible with BERT learning
"""

from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer


class BertProteinSequenceTokenizer(EsmSequenceTokenizer):
    """
    ESM tokenizer modified for BERT training compatibility

    This class wraps the original EsmSequenceTokenizer to make it compatible
    with BERT training by overriding model_input_names to use standard BERT format
    """

    # Override model input names to use standard BERT format
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure pad_token is set for BERT compatibility
        if not hasattr(self, "pad_token") or self.pad_token is None:
            self.pad_token = self.unk_token
            self.pad_token_id = self.unk_token_id


def create_bert_protein_tokenizer(**kwargs) -> BertProteinSequenceTokenizer:
    """
    Create a BERT-compatible protein sequence tokenizer

    Returns:
        BertProteinSequenceTokenizer instance
    """
    return BertProteinSequenceTokenizer(**kwargs)
