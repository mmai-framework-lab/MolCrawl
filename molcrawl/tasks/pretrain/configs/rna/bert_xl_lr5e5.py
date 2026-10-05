# rna BERT xl -- learning rate 5e-5. Separating two explanations of the same result.
#
# All three of 1e-4 / 1.4e-4 / 2e-4 collapsed, and none of them ever learned: their
# best values were 9.07 to 9.37 against a baseline of 9.372, and by step 16,000 they
# sat at 10.6 while the other sizes reach 2.7 well before that. The shape is large's
# 1e-3, which also never learned, rather than the shape of a run that learned and
# then broke.
#
# Two things would produce that, and the grid as placed cannot tell them apart.
#
#   1. The boundary is below 1e-4. rna's boundary falls by about 1.8 per step of
#      size, which put xl near 1.4e-4 -- but if it falls faster than that at this
#      size, the whole span sat above it.
#   2. Something at this size is unstable regardless of the rate. large runs 1e-4
#      without trouble, so a boundary that lands between large's 1e-4 and xl's is a
#      drop of more than 2 in one step, which no other pair in rna shows.
#
# 5e-5 is half of the span's lowest point. If it learns, the boundary is simply
# lower than the span and reading 1 holds. If it collapses the same way, the rate is
# not what is wrong and reading 2 has to be taken up -- precision, or the shape at
# 1.5B, rather than the schedule.
#
# Everything else comes from bert_xl.py, so the comparison is against the three
# points it is meant to explain and not against a differently configured run.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_xl import *  # noqa: F401,F403

learning_rate = 5e-5
seed = 42
model_path = get_bert_output_path("rna", "xl") + "-lr5e5"
