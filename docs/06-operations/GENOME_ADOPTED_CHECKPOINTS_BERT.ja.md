# genome BERT の採用 checkpoint

確定: 2026-10-30 / 基本: 9 エポックの系列（`bert-small-<subset>-w1026`, 21 subset, bert-small,
窓 1,026, グローバルバッチ 2,560, LR 1e-4, `max_steps` は subset ごとに 122,684〜163,900）

この run 群を下流（ClinVar ゼロショット / 線形プローブ / GPT-2 との比較）で使うときの取り扱い。
GPT-2 側は `GENOME_ADOPTED_CHECKPOINTS_GPT2.ja.md`。

---

## 9 エポック・窓 1,026 を基本とする理由

**GPT-2 と step 数が揃っている。** 21 subset すべてで `max_steps` が 9 エポック GPT-2
（`gpt2-small-<subset>-ep9`）の `max_iters` と一致する（2026-10-30 に 21/21 実測）。
計算量を揃えた組はこの対である。

**窓 1,026 は学習時の形である。** `[CLS]` + 1,024 塩基 + `[SEP]` で、GPT-2 の 1,024 と
塩基の数が揃う（`workflows/genome-bert-train.sbatch`）。

**到達した損失の高い低いは、選ぶ理由にしない。**

## 選択基準

**`eval_loss_mask` が最小の保存 checkpoint を採用する。同値なら早い step を採る。**

`eval_loss` ではない。全体値は「正解が入力に残る 10%」に薄められ、3 水準の差が見えにくい。
判定に `eval_loss_mask` を使うと決めた以上、checkpoint の選択も同じ基準に揃える。

- 採用 checkpoint のパス:
  `learning_source_genome_runs/bert-small-<subset>-w1026/checkpoint-<採用step>/`
- 採用 step は 99,000〜153,000、中央値 142,000。run 長に対する位置は 69.2〜100.0%、
  中央値 94.3%（`eukaryote_matched_random_seed1` の 69.2% だけが離れている）
- eval は 1,000 step ごと、保存も 1,000 step ごとなので、**eval 点と保存点が一致する。**
  3 エポックの系列にあった「保存されていない eval 点のほうが良い」代償はこの系列には無い
- 21 subset すべてで採用 checkpoint がディスク上に実在する（2026-10-30 検証）
- 下流の出力に記録された checkpoint は 21/21 とも採用 checkpoint である
  （2026-10-30 に 7 系列 × 21 = 147 通りを照合）

**`metric_for_best_model` が追う指標と採用基準が一致している。** HF が保護するのは
`metric_for_best_model` の best だけなので、両者が違えば採用 checkpoint が FIFO 削除で
消えうる。この 21 本は 2026-10-28 の全数照合で一致した。

## 最終モデルは使わない

**最終 step の重みは採用しない。** 反転幅（最終値 − best）は 0.0000〜+0.0039、
中央値 +0.0015 で、21 本のうち 20 本が最終値 > best である
（`eukaryote_matched_random_seed6` だけ best が最終 eval 点と同じ step）。

**反転が小さいことは最終値が安定していることの証明にはならない。** BERT では同一条件の
再走による再現性測定をしていない。GPT-2 では「best は安定・最終値は不安定」が実測されており
（9 エポックの系列は中央値 +0.1800・最大 +0.5958）、測っていない側で例外を作る理由がない。

## subset 間で best val を並べない

**subset は別データで、難易度差を含む。絶対値の横並びは比較として成立しない。**

採用値は 1.0693〜1.2752 で幅は 0.2059 あるが、これは性能差ではない。
**比較は同一 subset 内に限ること。**

## GPT-2 の値と比較しない

GPT-2 は CLM（左文脈・全位置）、BERT は MLM（両側文脈・マスク位置のみ）である。
語彙がどちらも 10 なので数値のレンジが重なるが、**同じ物差しではない。**
検証集合の取り方も違う（BERT は valid の先頭 10,000 行で固定、GPT-2 は毎回 64,000 行を
重複ありで抽選）。性能比較には同一の下流タスク評価が要る。

## 3 エポック・窓 512 の系列（`bert-small-<subset>`, 系列名 `bert-base`）

**2026-08-25 に採用していたが、現在は使わない。** 2026-10-30 の判断書により、
9 エポック・窓 1,026 を基本とする。

出力と checkpoint は消していない。採用 step は 82,000〜109,000、
per-run の全項目は `_results/bert21-per-run.csv`、eval 系列そのものは
`_results/bert21-eval-series.csv`（21,914 点）。4 指標（`eval_loss` / `eval_loss_mask` /
`eval_loss_copy` / `eval_loss_random`）を両ファイルに残してある。

この系列について記録してあった事実は次のとおりで、取り消していない。

- 採用できなかった run は無い（21/21 で `eval_loss_mask` 最小の保存 checkpoint を採用）
- **`metric_for_best_model` は `eval_loss` だった**ため、HF の印と採用が 9 本で一致しない。
  採用 checkpoint が残っていた理由は、HF が保護していた 10 本と、
  **たまたま `save_total_limit=5` の直近に入っていた 11 本**である。best が中盤に来る学習なら
  消えていた
- eval は 100 step ごと、保存は 1,000 step ごとだったので、採用 checkpoint と
  「全 eval 点を通じた真の最良」の差は平均 +0.0014・最大 +0.0029。一致したのは 2/21 本のみ
- 反転幅は +0.0012〜+0.0057、中央値 +0.0026、21 本すべてで最終値 > best
- **`trainer_state.json` の `best_model_checkpoint` は絶対パスで、105 個のファイルが
  存在しないパスを指している**（2026-08-26 に worktree から本体 checkout へ移動したため）。
  resume と本書と CSV には影響しない（step 番号で記録しているため）。
  このフィールドを直接読まず、採用 step は `bert21-per-run.csv` の `adopted_step` 列を見ること

## 記録項目

9 エポックの per-run 値（best / best@step / 最終値 / 反転幅 / 処理トークン数）は、
リポジトリ外の `report/genome-loss-curves/genome-loss-per-run.tsv` の `series` 列が
`bert-w1026` の 21 行にある。eval 系列そのものは同じディレクトリの
`genome-loss-curves.tsv`（2026-10-20 時点）。

## 最終 checkpoint を指しているコード

次の 3 本は、旧世代の run 木（`learning_source_20260617/genome_sequence/`）の
**最終 step** を step 番号で固定している。この 3 本はいまの図表のどれにも使っていない。
2026-10-30 に各ファイルの冒頭へその旨を書いた。

- `scripts/run_clinvar_eval_subsets.py`: bert `checkpoint-60000` / gpt2 `checkpoint-50000`
- `scripts/run_clinvar_finetune_bert_subsets.py`: `checkpoint-60000`
- `scripts/run_clinvar_finetune_gpt2_subsets.py`: `checkpoint-50000`

現行の 2 経路（`scripts/clinvar_zeroshot_bert_rescore.py` と
`scripts/clinvar_linear_probe.py`）は、採用 checkpoint が解決できないとき
最大 step に落ちていたが、2026-10-30 に**止める**ようにした。採った step は出力に書く。
