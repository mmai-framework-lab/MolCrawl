"""mol_nl BERT xl at 8.5e-5 — tighter gradient clipping

8.5e-5 is the lowest rate that diverged: it reached 3.7699 at step 1,900 and was above the
unigram floor again by 2,100. The four rates below it (6e-5 down to 2.2e-5) never diverged
and never learned either -- 2.96 to 3.21 at step 15,000, where large reads 0.0691. So xl
has no rate that both holds and trains, and the suspect is the depth: 48 layers against
large's 36, in a post-LN BertModel.

The clip goes from 1.0 to 0.5. If the turn is driven by a
few large updates rather than by the rate itself, halving what any step may do is what
fixes it.

One thing changes from the stopped runs. Everything else -- 24,000 steps, 8 x 80 on 4 GPUs,
seed 42, the fixed evaluation mask -- is theirs.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_xl_24k import *  # noqa: F401,F403

learning_rate = 8.5e-5

max_grad_norm = 0.5
