# genome GPT-2 の採用 checkpoint

確定: 2026-10-30 / 基本: 9 エポックの系列（`gpt2-small-<subset>-ep9`, 21 subset, gpt2-small,
グローバルバッチ 2,560, `max_iters` は subset ごと）

この run 群を下流（detector / genome BERT 比較 / §7 評価）で使うときの取り扱い。
BERT 側は `GENOME_ADOPTED_CHECKPOINTS_BERT.ja.md`。

---

## 9 エポックを基本とする理由

**BERT と step 数が揃っている。** 21 subset すべてで BERT (`bert-small-<subset>-w1026`) の
`max_steps` と 9 エポック GPT-2 の `max_iters` が一致する（2026-10-30 に 21/21 実測、
122,684〜163,900）。計算量を揃えた組はこの対である。

**到達した損失の高い低いは、選ぶ理由にしない。**

## 選択基準

**val loss が最小の保存 checkpoint を採用する。同値なら早い step を採る。**

- nanoGPT は val loss が改善したときだけ run 直下の `ckpt.pt` を書き換えるので、
  このファイルが best-val の重みである
- 21 subset すべてで `ckpt.pt` と `checkpoint-<best_step>/` の両方がディスク上に実在する
  （2026-10-30 検証）
- 採用 checkpoint のパス: `learning_source_genome_runs/gpt2-small-<subset>-ep9/ckpt.pt`
- best@step は 19,000〜105,000、中央値 28,000。`mammal_centered` の 105,000 だけが離れている

`global_random_seed4` は `ckpt.pt` が step 25,000、per-run 表の best@step が 23,000 である。
両 step の val loss は記録精度（小数 6 桁）で同値の 1.224900 で、nanoGPT は 25,000 を、
per-run 表は「同値なら早い step」の規則で 23,000 を採っている。どちらも最終 step ではない。

## 最終モデルは使わない

**最終 step の重みは採用しない。** 9 エポックの系列は 21 本すべてで最終値 > best val であり、
その差は中央値 +0.1800、最大 +0.5958（`global_random_seed3`）である。
3 エポックの系列は中央値 +0.0091・最大 +0.1259 なので、中央値で 20 倍ほど大きい。
使う場面が生じたら個別に相談すること。

`mammal_centered` だけは差が +0.0046 で、best@step も 105,000 と遅い。
この 1 本だけで系列の形を読まないこと。

## subset 間で best val を並べない

**subset は別データで、難易度差を含む。絶対値の横並びは比較として成立しない。**

`mammal_centered` の best val 1.0875 は 21 本中 1 位だが、これは難易度が低いことも寄与している。
**比較は同一 subset 内に限ること。**

## 3 エポックの系列（`gpt2-small-<subset>`）

**2026-08-21 に採用していたが、現在は使わない。** 2026-10-30 の判断書により、
図表にも本文にも出さない。

出力と checkpoint は消していない。run 直下の `ckpt.pt` は 21/21 とも best-val の重みであり
（2026-10-30 に mtime 照合）、採用 step は 27,000〜50,000 である。
per-run の全項目は `_results/b21-per-run.csv`、val 系列そのものは `_results/b21-val-series.csv`。

この系列について記録してあった事実は次のとおりで、取り消していない。

- **最終値 > best val は 21 本すべて**。反転幅は中央値 +0.0091、最大 +0.1259
  （`global_random_seed3`）、`mammal_centered` の +0.0760 は 3 番目に大きい。
  旧版は「21 本中 10 本が最終値 > best val」と書いていたが、これは反転幅 0.01 以上が
  10 本という意味で、2026-10-30 に per-run 表から数え直して訂正した
- 21 本中 18 本が途中で 0.02 以上の val スパイクを起こし、うち 8 本は完全に回復した。
  最終時点で反転が残るかは、スパイクの大きさよりも発生時期に依存する
  （順位相関: 大きさ +0.218 / 発生 step +0.515 / 残り step −0.427、n=21。いずれも中程度）
- `mammal_centered` は step 52,000 でスパイクし、回復する前に run が終わっている。
  採用 checkpoint（step 34,000）はスパイク発生前なので影響を受けない

## 記録項目

9 エポックの per-run 値（best / best@step / 最終値 / 反転幅 / 処理トークン数）は、
リポジトリ外の `report/genome-loss-curves/genome-loss-per-run.tsv` の `series` 列が
`gpt2-ep9` の 21 行にある。val 系列そのものは同じディレクトリの
`genome-loss-curves.tsv`（2026-10-20 時点）。

## 最終 checkpoint を指しているコード

次の 3 本は、旧世代の run 木（`learning_source_20260617/genome_sequence/`）の
**最終 step** を step 番号で固定している。この 3 本はいまの図表のどれにも使っていない。

- `scripts/run_clinvar_eval_subsets.py`: bert `checkpoint-60000` / gpt2 `checkpoint-50000`
- `scripts/run_clinvar_finetune_bert_subsets.py`: `checkpoint-60000`
- `scripts/run_clinvar_finetune_gpt2_subsets.py`: `checkpoint-50000`

旧世代の 21 run では 60000 / 50000 がそれぞれ最大の checkpoint である（2026-10-30 実測）。
2026-10-30 に各ファイルの冒頭へその旨を書いた。

現行の 2 経路（`scripts/clinvar_zeroshot_bert_rescore.py` と
`scripts/clinvar_linear_probe.py`）は、採用 checkpoint が解決できないとき
最大 step に落ちていたが、2026-10-30 に**止める**ようにした。採った step は出力に書く。
