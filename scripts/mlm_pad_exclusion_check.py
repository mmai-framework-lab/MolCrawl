"""pad が MLM のマスク選択と損失から除外されているかを実測する。

RNA では pad と細胞境界が同じ pad id なので、除外されていれば境界も自動的に除外される。
除外されていなければ、マスクの 2 割が詰め物の予測に使われる。
これは RNA 固有ではなく、BERT を使う 5 モダリティすべてに効く。
"""
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer

from molcrawl.models._collators import ambiguous_tokens_for_modality, make_mlm_collator
def LOG(*a): print(*a, flush=True)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tokenizer-dir", required=True,
                help="調べる tokenizer のディレクトリ。モダリティごとに違う")
ap.add_argument("--modality", default=None,
                help="曖昧トークンの表を引くための名前（rna/protein_sequence など）")
ap.add_argument("--pad-id", type=int, default=0,
                help="データ側が詰めに使っている id。RNA では細胞境界と同じ 0")
A = ap.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer_dir)
LOG("=== 1. tokenizer が返す特殊トークン id ===")
for name in ["pad_token", "mask_token", "unk_token", "sep_token", "cls_token"]:
    t = getattr(tok, name, None)
    i = getattr(tok, name + "_id", None)
    LOG(f"  {name:12s} {str(t):8s} -> {i}")
LOG(f"  all_special_ids: {sorted(tok.all_special_ids)}")
LOG(f"  vocab_size={tok.vocab_size} len(tok)={len(tok)}")

LOG("\n=== 2. pad id は特殊トークン扱いか（マスク選択から外れるか）===")
ids = [0, 1, 2, 3, 100, 25426, 25427, 25428]
m = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)
for i, f in zip(ids, m):
    LOG(f"  id {i:6d}: special={bool(f)}")

LOG("\n=== 3. collator を実際に通す（学習と同じ経路）===")
coll = make_mlm_collator(tok, ambiguous_tokens=(ambiguous_tokens_for_modality(A.modality) if A.modality else []) or [],
                         mlm_probability=0.2)
LOG(f"  collator: {type(coll).__name__}")
rng = np.random.default_rng(0)
# 前半は普通のトークン、後半は pad id（RNA では pad でも細胞境界でもある）
L = 64
hi = max(tok.vocab_size - 1, 8)
seq = list(rng.integers(2, hi, size=L - 24)) + [A.pad_id] * 24
batch = [{"input_ids": [int(x) for x in seq]} for _ in range(400)]
torch.manual_seed(0)
out = coll(batch)
inp, lab = out["input_ids"], out["labels"]
sel = lab != -100
z = torch.tensor(seq) == A.pad_id
LOG(f"  系列長 {L}（うち pad id が {int(z.sum())} 個）x {len(batch)} 本")
LOG(f"  マスク選択された位置: {int(sel.sum()):,} / {sel.numel():,} "
    f"({sel.float().mean()*100:.1f}%)")
z_b = z.unsqueeze(0).expand_as(sel)
LOG(f"    うち pad id の位置: {int((sel & z_b).sum()):,}")
LOG(f"    うち pad id 以外  : {int((sel & ~z_b).sum()):,}")
den = int((~z).sum()) * len(batch)
LOG(f"  pad id 以外に対する選択率: {int((sel & ~z_b).sum())/max(den,1)*100:.1f}%"
    "  （mlm_probability=0.2 と比べる）")
LOG("")
if int((sel & z_b).sum()) == 0:
    LOG("RESULT pad(=境界) は マスク選択から除外されている")
    LOG("RESULT 損失にも入らない（labels=-100 の位置は損失に寄与しない）")
else:
    LOG("RESULT pad(=境界) が マスク選択に入っている — マスク予算が詰め物に使われる")
