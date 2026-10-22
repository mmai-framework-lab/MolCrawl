"""mol_nl BERT medium 24,000 step・学習率 3e-4・seed 1。

medium は 3 サイズのうち唯一、代表の学習率に seed が 1 本しかない。seed 3 本が揃うのは small
1e-3 と large 1.7e-4 で、medium の 3 本は代表ではない 4.1e-4 にあり、そのうち seed 17 は step 13,000 で
発散している。このため 2026-10-19 の報告では medium の行だけ seed 幅が書けず、medium と large の順位も
幅つきでは言えていない。

この 3 本は同時に走らせる必要がある。PR #307 で eval_loss_mask が全ランク合算に直り、それ以前の値は
rank 0 の 1/4 シャードのものだからである。既存の seed 42 の run は古い定義で測られており、新しい定義で
測った seed 1・17 と混ぜた幅は、幅ではなく定義の差を測ることになる。

seed 以外はすべて bert_medium_24k_lr3e4 から import する。

未投入。2026-10-22 時点で、3 本そろえて走らせるかどうかの判断待ち。
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium_24k_lr3e4 import *  # noqa: F401,F403

seed = 1
