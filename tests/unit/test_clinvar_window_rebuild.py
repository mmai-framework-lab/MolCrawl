"""How the 1,024-base ClinVar window is cut, and what it refuses to emit.

The models were pretrained on 1,024-token windows and the shipped evaluation
table carries 129 bases, so scoring ran on an eighth of the context the models
were built for. The rebuild changes the length and nothing else.

512 either side would be 1,025 and overrun both GPT-2's 1,024-token block and
BERT's [CLS] + 1,024 + [SEP]. The window is therefore asymmetric -- 512 before,
the variant, 511 after -- so it fits the models exactly and the variant sits one
base left of geometric centre.
"""
import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "rebuild_clinvar_windows.py"
_spec = importlib.util.spec_from_file_location("_rebuild_clinvar", _SRC)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def test_the_window_is_exactly_the_models_context():
    """1,024 fits GPT-2's block_size and BERT's 1,026 with CLS and SEP."""
    assert rb.LEFT + 1 + rb.RIGHT == 1024


def test_the_variant_sits_513th_not_at_geometric_centre():
    """An even-length window has no centre; the offset is chosen, not incidental."""
    assert rb.CENTRE == 512
    assert rb.LEFT != rb.RIGHT


def test_the_slice_offsets_put_the_variant_where_centre_says():
    """start0 = pos-1-LEFT, end0 = pos+RIGHT must place pos at index CENTRE."""
    pos = 100_000
    start0 = pos - 1 - rb.LEFT
    end0 = pos + rb.RIGHT

    assert end0 - start0 == 1024
    assert (pos - 1) - start0 == rb.CENTRE
