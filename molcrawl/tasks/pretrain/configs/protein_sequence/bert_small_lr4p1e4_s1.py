# protein BERT seed-variance — small, lr 4.1e-4, seed 1.
# protein-order-2026-10-01 §5: the collapse boundary (small 3e-4 survived / 4.1e-4 collapsed
# at seed 42) is tested for seed dependence. This arm reruns the collapsed minimum LR at a
# different seed; if it no longer collapses, the boundary is seed-dependent. seed 1 and 17
# are the variance seeds (boss directive 2026-09-30). main.py sets data_seed=seed, so this
# is seed=data_seed=1. Everything else (adam_beta2=0.999, warmup 3,353, max_steps 33,531,
# 8x80 eff 2,560, bf16) comes from base bert_small.py. Distinct out path per arm.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_small import *  # noqa: F401,F403

learning_rate = 4.1e-4
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 1                    # variance seed (data_seed=seed via main.py)
model_path = get_bert_output_path("protein_sequence", "small") + "-lr4p1e4-s1"
