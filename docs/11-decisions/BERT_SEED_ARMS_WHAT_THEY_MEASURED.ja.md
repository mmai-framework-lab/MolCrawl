# BERT の seed arm が測ったもの

2026-10-29（`seed=seed` を戻す）と 2026-11-10（6 本は取り直さない）で確定。BERT の
seed arm ごとに、宣言した seed が `transformers` に届いたかどうかを記録する。ペアから
引いた幅を正しい名前で呼べるようにするための文書である。英語版は
`BERT_SEED_ARMS_WHAT_THEY_MEASURED.md`。

この文書の要点。**「seed の幅」として報告してきた 4 つの幅のうち、3 つは seed の幅で
あり、1 つは違う。** 4 つとも 0.0012 から 0.0015 の同じ大きさなので、1 つのものとして
読めてしまっていた。

---

## seed が `transformers` に届いた経路（時期別）

| 時期 | `TrainingArguments` が受け取った値 | commit |
|---|---|---|
| 2026-08-02 まで | `seed=` なし。`args.seed` はライブラリ既定の 42 | -- |
| 2026-08-03 から 2026-10-01 | `seed=seed` と `data_seed=seed` | `05ee336d` |
| 2026-10-02 から 2026-10-29 | `data_seed=seed` のみ。`args.seed` は 42 に落ちた | `0f3e7067` |
| 2026-10-30 以降 | モデル生成より前に `set_seed(int(seed))`、および `data_seed=seed` | `97069327`（PR #340） |

3 番目の時期が問題になる理由は、`transformers` 4.45.1 について実測した 2 点である。

`TrainingArguments.seed` の既定値は **42** である。`seed=` を落としても未設定にはならず、
本プロジェクトが標準としていたその値が黙って入る。

`trainer.py` に **`data_seed` の参照は 1 つも無い**（ファイル内 0 件）。データ順は
`set_seed(self.args.seed)` に従い、これは `Trainer.__init__`（`trainer.py:406`）と
学習ループ（`:2010`）が呼ぶ。したがって `args.seed` が 42 のまま config が
`data_seed = 1` を宣言しても、何も変わらない。

`tests/unit/test_bert_seed_reaches_transformers.py` が両方を留めている。2 番目は
`transformers` を上げて `data_seed` が読まれ始めたら落ちる。

## 宣言した seed が `transformers` に届かなかった 6 本

BERT 324 本の全数走査（2026-11-10、`scripts/bert_seed_inventory.py`）。
`seed_agrees=no` は 6 行で、3 ペアである。

| run | config の指定 | manifest の `seed.seed` | `seed.data_seed` | job | commit |
|---|---|---|---|---|---|
| `bert_medium_24k_lr3e4_seed1` | 1 | 42 | 1 | 199853 | `4a3d9f4` |
| `bert_medium_24k_lr3e4_seed17` | 17 | 42 | 17 | 199855 | `4a3d9f4` |
| `protein_sequence-large-lr5e5-s1` | 1 | 42 | 1 | 152889 | `fd1dbb3` |
| `protein_sequence-large-lr5e5-s17` | 17 | 42 | 17 | 152893 | `fd1dbb3` |
| `protein_sequence-large-lr7e5-s1` | 1 | 42 | 1 | 173558 | `fd1dbb3` |
| `protein_sequence-large-lr7e5-s17` | 17 | 42 | 17 | 173562 | `fd1dbb3` |

6 本とも 3 番目の時期（2026-10-06 から 2026-10-22）に走っている。他の seed ペアは
すべて 2026-10-02 より前に走り、manifest の値は宣言どおりである。GPT-2 は影響を
受けていない（`train.py` が自分で `torch.manual_seed(seed + ddp_rank)` を呼ぶ。
同じ走査で 343 本中 食い違い 0 件）。

## 報告してきた幅が何を測ったか

`eval_loss_mask`、最後の 10 評価の平均。各 run の `trainer_state.json` から実測。

| ペア | arm ごとの `args.seed` | 値 | 幅 | 変わったもの |
|---|---|---|---|---|
| molnl small 1e-3 | 42 / 1 / 17 | 0.059614 / 0.058306 / 0.058521 | **0.0013** | seed |
| molnl large 1.7e-4 | 42 / 1 / 17 | 0.057465 / 0.058699 / 0.058008 | **0.0012** | seed |
| molnl medium 3e-4 | 42 / 42 / 42 | 0.057386 / 0.055869 / 0.056038 | **0.0015** | **宣言したものは何も変わっていない** |
| protein large 5e-5 | 42 / 42 | 2.534990 / 2.540291 | **0.0053** | **宣言したものは何も変わっていない** |
| protein large 7e-5 | 42 / 42 | -- | -- | 33,531 step のうち 7,600 と 7,700 で打ち切り |

molnl medium の 3 本は `bert_medium_24k_lr3e4_s42r`・`_seed1`・`_seed17` で、いずれも
2026-10-22 に `4a3d9f4` から投入され、評価の定義も同じである。

protein 5e-5 の 3 本目 `protein_sequence-large-lr5e5`（seed 42）は 2 本と比べられない。
33,531 step のうち 30,000 で止まり、best は step 5,400 の 2.7526 である。上の幅は
s1 と s17 の間だけを取っている。

protein 7e-5 の 3 本は早期に打ち切られたので、幅は出ない。

**違う seed を宣言して同じ値を受け取った 2 本が、それでも違う値になる。** seed 以外にも
緩んだままのものがあったからである。3 番目の時期には、重みを作る前に種付けするものが
無かった。モデルは `TrainingArguments` より前に構築されるので、プロセス内で最初の
`set_seed` は `Trainer.__init__` の中、つまりモデルより後だった。初期の重みは種付け
されていない生成器から出ている。PR #340 が `set_seed` をモデルより前に移した変更である。

実測（job 312632、計算ノード）。種付け前に小さな `BertForMaskedLM` を作る 2 プロセスは
`torch.initial_seed()` が 1479681010697427743 と 14957004836947191761、重みの SHA-256 が
`e61940822dd90e9c` と `316b7c1ce34f59e0` で一致しない。`set_seed(42)` をモデルより前に
置いた 2 プロセスは、どちらも `initial_seed` 42・重み `42ff42a10f839006` で一致する。

## 資料に書く文

**molnl medium の幅 0.0015 は seed の幅ではない。宣言した条件が同一の run どうしの幅
である。** small の 0.0013 と large の 0.0012 は seed の幅である（arm が 1 と 17 を
受け取っている）。

資料で 3 つを「seed の幅は 0.0012 から 0.0015」とまとめてはならない。3 つが揃って
支持するのはもっと狭く、それでも役に立つ事実である。この水準では、seed が違う run
どうしと、何も違わない run どうしが同じだけ離れる。したがってこの大きさの差は、
どちらの向きにも順位を与えない。

## 取り直さない

3 ペアは取り直さない。1,500 GPU-h であり、取り直して分かることは測定済みの事実と
同じである（2026-10-29 §4 の決定、2026-11-10 §3.3 で確認）。

## 値の置き場所

- BERT 324 本の run ごとの seed: `report/rt1-bert-seed-inventory-2026-11-10.tsv` の
  `config_seed`・`manifest_seed`・`seed_agrees` 列。
- run 条件の表は、run ごとに食い違いを
  `42（config は 1 を指定。効いていない）` として既に載せている:
  `report/run-conditions-20261102.tsv`。
- 上の幅: 各 run の `trainer_state.json`、および molnl の集計
  `report/rt1-molnl-summary-runs.tsv` の `window_mean` 列。
- `learning_source*` には手を入れていない。走った値は manifest にある。
