# rna BERT xl -- the base the three learning-rate points import.
# rna-followup-order-2026-09-30 §5. rna had no xl config; the shape itself already
# exists in models/bert/main.py (hidden 1600, 48 layers, 25 heads, intermediate 6400,
# matching GPT-2 XL's n_embd/n_layer/n_head). What was missing was a config naming it.
#
# Everything but the size, the output directory and the rate comes from bert_large.py,
# imported rather than copied, so max_steps 120,960 and warmup_steps 12,096 (9 epochs,
# 10 %), 8 x 80 on 4 GPUs for an effective 2,560, bf16 with four dataloader workers and
# pinned buffers, and document masking with boundary id 0 stay identical to the sizes
# this one will be compared against.
#
# Counted from the shape: 1,561.2M parameters in total, 1,478.1M of them outside the
# embeddings. The rna vocabulary is 25,432, so the embeddings are 83.1M here against
# 20.3M at small -- which is why sizes are placed on a compute axis by the
# non-embedding count rather than the total.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_large import *  # noqa: F401,F403

# No annotations on these three: the star import above already bound the names, and
# re-annotating a name mypy has seen is a redefinition. protein_sequence's and
# compounds' bert_xl.py assign them the same way.
model_size = "xl"
model_path = get_bert_output_path("rna", model_size)

# The three grid points override this. It is set here so that running the base
# directly does not silently inherit large's env-driven rate.
learning_rate = 1.4e-4
