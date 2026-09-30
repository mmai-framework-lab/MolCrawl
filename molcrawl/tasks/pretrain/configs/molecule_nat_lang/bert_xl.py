"""mol_nl BERT xl -- the size above large.

models/bert/main.py already carries the shape (hidden 1,600, 48 layers, 25 heads,
intermediate 6,400); what was missing was a config for this modality pointing at it.
compounds has had one since 2026-09. 12 x 48 x 1,600^2 is 1.47 G non-embedding
parameters, 2.56 times large -- the run itself is what the count gets read from.

Everything else comes from bert_large.py. The batch shape is unchanged at 8 x 80, which
is 2,560 sequences on 4 GPUs; if that does not fit in memory at this size the product
times the GPU count is what has to be held, not the two numbers.
"""

from molcrawl.core.paths import get_bert_output_path  # noqa: F401
from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large import *  # noqa: F401,F403

model_size = "xl"
model_path = get_bert_output_path("molecule_nat_lang", model_size)
