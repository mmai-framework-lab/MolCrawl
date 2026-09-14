"""境界 id の走査が、無い場合に止まり、ある場合に数えられることを確かめる。

document masking が黙って no-op になる経路は、protein・molecule_nat_lang・RNA で
3 度見逃されている。いずれも「設定は正しく見えるのに効いていない」形だった。
"""
import pytest

from molcrawl.models.bert.main import _scan_boundary_id


class _Rows:
    """datasets.Dataset の最小限の代役（__len__ と [i]["input_ids"]）。"""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return {"input_ids": self._rows[i]}


def test_counts_boundaries_and_blocks():
    # 1 行 8 トークン、境界 id 0 が 2 個ずつ
    rows = _Rows([[5, 6, 0, 7, 8, 0, 9, 10]] * 4)
    st = _scan_boundary_id(rows, 0)
    assert st["rows"] == 4
    assert st["positions"] == 32
    assert st["count"] == 8
    assert st["per_block"] == pytest.approx(2.0)
    assert st["rate"] == pytest.approx(8 / 32)


def test_absent_boundary_is_zero_not_an_error():
    """走査そのものは投げない。無いと分かることが仕事で、判断は呼び手が下す。"""
    rows = _Rows([[5, 6, 7, 8]] * 3)
    st = _scan_boundary_id(rows, 25428)
    assert st["count"] == 0
    assert st["rows"] == 3


def test_sample_is_capped():
    rows = _Rows([[0, 1, 2]] * 1000)
    st = _scan_boundary_id(rows, 0, n_rows=10)
    assert st["rows"] == 10
    assert st["count"] == 10


def test_unreadable_dataset_reports_zero_rows():
    """長さを持たない相手でも落とさない。manifest の書き出しを巻き添えにしない。"""
    st = _scan_boundary_id(object(), 0)
    assert st == {"rows": 0, "positions": 0, "count": 0, "rate": 0.0, "per_block": 0.0}


def test_rna_boundary_and_sep_disagree():
    """RNA で実際に起きた形: 区切りは 0、tokenizer の sep_token_id は語彙の外。

    同じ id なら走査は同じ答えを返す。違う id なら片方は 0 になる -- それが
    「設定は正しく見えるのに効いていない」の正体で、停止の根拠になる。
    """
    rows = _Rows([[12, 34, 0, 56, 78]] * 5)
    assert _scan_boundary_id(rows, 0)["count"] == 5
    assert _scan_boundary_id(rows, 25428)["count"] == 0
