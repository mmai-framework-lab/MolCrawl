# protein BERT LR grid — large, lr 7e-5, seed 17 (boundary variance, upper edge).
# protein-order-2026-10-05 §4: large 5e-5 collapsed at seed 42 but survived at seed 1/17,
# so the boundary moves with the seed. This arm places 7e-5 at variance seed 17. main.py sets
# data_seed=seed. Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353,
# max_steps 33,531, 8x80 (eff 2,560), bf16 from bert_large. Distinct out path per arm.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

learning_rate = 7e-5
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 17                   # variance seed (data_seed=seed via main.py)
model_path = get_bert_output_path("protein_sequence", "large") + "-lr7e5-s17"
