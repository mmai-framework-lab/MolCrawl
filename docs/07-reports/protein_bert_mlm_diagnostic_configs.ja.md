# protein BERT の MLM 診断 config（2026-08〜2026-09）

main checkout で git 管理外のまま置かれ、protein BERT の 11 本の学習を走らせた 6 ファイル。
これがリポジトリに入っている理由と、各ファイルが何を生んだかの対応を記録する。

## なぜ未追跡だったか、なぜ今コミットするか

protein BERT small が unigram 周辺解に落ちる件（本番 run 24870）の原因調査で書かれたもの。
いずれも `bert_small.py` に対して 1 点だけ変える形なので、作って投入して手で直す、を
コミットせずに繰り返した。2026-10-30 の未追跡ファイル整理で見つかり、他の 62 件は退避か
コミットに回したが、この 6 件だけは扱いを判断待ちとして残置した。

2026-11-10 の seed 一律 42 で漏れが表に出た。検査は git ではなく作業ディレクトリを走査する
ため、ツリー内で 42 でない config はこの 5 本だけになる。**run を投入した現物をバージョン
管理下に置くのがこのコミットの目的**で、seed をどうするかは別の判断として末尾に書く。

**現状のままコミットしており、走った時点のままではない。** job log は config 全文を rank ごとに
echo しているので、それと突き合わせると差が 3 つある。

| 差 | 対象 | log の echo | ここでコミットした内容 |
|---|---|---|---|
| run 後に `save_on_improve = True` が追記 | config 5 本すべて | 無し | 有り |
| `per_device_eval_batch_size` | `caseA`・`caseC` | 8 | 160 |
| `gradient_accumulation_steps` は投入時の上書きだった | `bert_small_mlm015_diag.py` | `5 * 16`（投入時に 4 へ上書き） | `5 * 16` |

run が実際に使った本文はファイルではなく log を読む。log は `workflows/slurm-logs/`
（git 管理外・クラスタ上）。

## ファイル

| ファイル | `bert_small.py` に対する変更点 | sha256（コミットした写し） |
|---|---|---|
| `configs/protein_sequence/bert_small_mlm015_diag.py` | train mask 0.20 → **0.15**。旧形式の config（`adam_beta2` は GPT-2 仕様の 0.95 のまま） | `22e4bdbc507e92d0…` |
| `configs/protein_sequence/bert_small_esmopt_diag.py` | **`adam_beta2` 0.95 → 0.999 と `warmup_steps` 200 → 2000 だけ**（ESM/BERT 標準の最適化） | `7271268bb0857ee2…` |
| `configs/protein_sequence/bert_small_esmopt_lr4e4_diag.py` | 上の 2 点＋ LR 3e-4 → **4e-4** | `e19f28f96113d266…` |
| `configs/protein_sequence/bert_small_esmopt_caseA_lr1e4.py` | 案 A：LR 3e-4 → **1e-4**、micro-batch 160×4 | `97ee8a33731ab98d…` |
| `configs/protein_sequence/bert_small_esmopt_caseC_mask040.py` | 案 C：train mask 0.20 → **0.40**、micro-batch 160×4 | `dc20860876e041cb…` |
| `models/esm2/main_bertswap_diag.py` | 休止中の esm2 入口の `EsmConfig`/`EsmForMaskedLM` を `BertConfig`/`BertForMaskedLM` に差し替えたもの（作業 1：同じ BERT モデルは esm2 の経路なら protein MLM を学習するか） | `f04d2b2a8c0b8f68…` |

## 各ファイルが生んだもの

| run | 由来 | job / 日付 | 当時のコード | 結果（best） |
|---|---|---|---|---|
| `runs/bert-small-mlm015-eb128` | `mlm015_diag` | 33280、2026-08-21 | 記録なし | `eval_loss` 2.6284 @1,800。20,000 のうち 14,000 で打ち切り |
| `runs/bert-small-warmup2k-mlm015-eb128` | `mlm015_diag` ＋ `warmup_steps=2000` | 33760、2026-08-21/22 | 記録なし | `eval_loss` 2.5266 @19,500。20,000 step 完走 rc=0 |
| `runs/_superseded/esm2script-bertmodel-diag-stopped40664` | `main_bertswap_diag.py` | 40664、2026-08-25 | 記録なし | 2.561（blended 指標・mask 0.15・eff-batch 128）。差し替え済 `SUP-BERT-BLENDED015` |
| `runs/bert-small-esmopt-diag` | `esmopt_diag` | 44835 → 53852 で再開、2026-08-25/27 | `18bb59c` → `32b4a01` | `eval_loss` 2.5649 @2,800。11,177 step（3 epoch）完走 rc=0 |
| `runs/bert-small-esmopt-lr4e4-diag` | `esmopt_lr4e4_diag` | 53853、2026-08-28/30 | `32b4a01` | `eval_loss` 2.5630 @2,600。11,177 step 完走 rc=0 |
| `runs/bert-caseA-lr1e4` | `caseA_lr1e4` | 77270（step 330 で停止）→ 77562、2026-09-02/05 | `bcd67bf` → `dbbc980` | `eval_loss_mask` 2.7264 @11,000。11,177 step 完走 rc=0 |
| `runs/bert-caseC-mask040` | `caseC_mask040` | 77271 → 77547、2026-09-02/05 | `bcd67bf` → `dbbc980` | `eval_loss_mask` 2.6813 @11,000。11,177 step 完走 rc=0 |
| `protein-bert-batchdiag/bert-batchdiag-eb128-lr4e4` | `caseA_lr1e4` ＋投入時上書き | 97111 → 97124、2026-09-11/12 | `3e0d4e7` | `eval_loss_mask` 2.7585 @9,000。37,266 step |
| `protein-bert-batchdiag/bert-batchdiag-eb640-lr4e4` | 同 | 97112 → 97125 | `3e0d4e7` | `eval_loss_mask` 2.6797 @7,200。7,453 step |
| `protein-bert-batchdiag/bert-batchdiag-eb2560-lr4e4` | 同 | 97113 → 97126 | `3e0d4e7` | `eval_loss_mask` 2.7720 @1,800。1,863 step |
| `protein-bert-batchdiag/bert-batchdiag-eb2560-lr3e4` | 同 | 97114 → 97127 | `3e0d4e7` | `eval_loss_mask` 2.7826 @1,800。1,863 step |

`runs/` は `learning_source_20260730_protein_uniref50/protein_sequence/runs/`。
`protein-bert-batchdiag/` はリポジトリ外かつどの `learning_source` ツリーでもない別ディレクトリ。

## 走った時の設定（生成物から実測）

各 run の最新 `checkpoint-*/training_args.bin` と、ある場合は `run_manifest.json` から読んだ値。

| run | LR | warmup | `adam_beta2` | micro×accum×GPU = eff-batch | train mask | best 判定指標 | seed |
|---|---|---|---|---|---|---|---|
| `bert-small-mlm015-eb128` | 3e-4 | 200 | 0.95 | 8×4×4 = 128 | 0.15 | `eval_loss` | 82 |
| `bert-small-warmup2k-mlm015-eb128` | 3e-4 | 2,000 | 0.95 | 8×4×4 = 128 | 0.15 | `eval_loss` | 82 |
| `bert-small-esmopt-diag` | 3e-4 | 2,000 | 0.999 | 8×80×4 = 2,560 | 0.20 | `eval_loss` | 82 |
| `bert-small-esmopt-lr4e4-diag` | 4e-4 | 2,000 | 0.999 | 8×80×4 = 2,560 | 0.20 | `eval_loss` | 82 |
| `bert-caseA-lr1e4` | 1e-4 | 2,000 | 0.999 | 160×4×4 = 2,560 | 0.20 | `eval_loss_mask` | 82 |
| `bert-caseC-mask040` | 3e-4 | 2,000 | 0.999 | 160×4×4 = 2,560 | **0.40** | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb128-lr4e4` | 4e-4 | 3,727 | 0.999 | 32×1×4 = 128 | 0.20 | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb640-lr4e4` | 4e-4 | 745 | 0.999 | 160×1×4 = 640 | 0.20 | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb2560-lr4e4` | 4e-4 | 186 | 0.999 | 160×4×4 = 2,560 | 0.20 | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb2560-lr3e4` | 3e-4 | 186 | 0.999 | 160×4×4 = 2,560 | 0.20 | `eval_loss_mask` | 82 |

全 run 共通：`lr_scheduler_type=linear`・`weight_decay=0.01`・`max_grad_norm=1.0`・
`adam_beta1=0.9`・`adam_epsilon=1e-8`・`seq_len=1024`・**`bf16=False`／`fp16=False`＝素の fp32**・
`eval_steps=100`・`save_steps=2000`・`DocumentMaskingCollator`（document masking 有効）・データは
`learning_source_20260730_protein_uniref50`（train 9,538,464 行・eval 10,000 行）。崩壊検出は
mlm015 の 2 本が grace 25,000、esmopt の 2 本が 15,000、案 A と案 C は
`degenerate_loss_threshold = None`（protein は検出 OFF・2026-08-26）。

## 各 run の記録の所在

| 項目 | config ファイル | job log | `training_args.bin` | `run_manifest.json` | 報告書 |
|---|---|---|---|---|---|
| seed・LR・warmup・`adam_beta2`・バッチ・`max_steps` | ○ | ○ | ○ | 11 本中 6 本 | ○ |
| 投入時の上書き | × | ○ | ○（解決後の値） | ○（`sources.*.from`） | ○ |
| **train mask 率** | ○ | ○ | **属性が無い** | **`objective.mlm_probability: null`** | ○ |
| コードの commit | — | 2026-08-25 以降 | × | ○ | ○ |
| eval 各点の loss | — | ○ | `trainer_state.json` | × | ○ |

案 C と mlm015 の 2 本が本題にしている mask 率だけは、どの run 出力にも残らない。残るのは
job log の config echo、このファイル群、そして `/data1/rkp00024/matsubara/report/` の報告書
（各 run 名が 27〜33 ファイルから参照され、案 C は「案 C（mask 0.40）2.695 対 案 A（LR 1e-4）2.738」
として記載）。

案 A・案 C・batchdiag 4 本の manifest は未追跡ファイル名を `run.git.dirty_files` に列挙しており、
「未追跡 config で走った」という事実自体も生成物側に残っている。

## ここでは決めない 2 件

1. **seed：2026-11-11 に決着**。この 5 本は run が使った値である `seed = 82` のままコミットし、
   直後のコミットでツリーの他と揃えて 42 にした。82 は各 run の `training_args.bin`・job log・
   上の表に残るので、config が名指しをやめても失われない。42 でない config は、seed 自体が
   主題である変動測定の 32 本だけになった。
2. **再投入**。どのファイルも、そのままでは当時の run を再現しない。mlm015 の 2 本は実効バッチが
   投入時の上書きで、上記 3 点の run 後の編集もファイルに入っている。実行用の本番 config ではなく、
   調査の記録として扱う。
