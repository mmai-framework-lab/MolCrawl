# 評価器 runnability audit — 2026-04-28

足固め適用済み (11/18) と残ブロッカー (7/18) の現状。

このドキュメントは 2 段階で更新された:

1. **当初の audit (午前)**: 7 評価器が足固め済み、残 11 のうち 4 件が「短い修正で実行可能」、7 件が「外部資格 / 閉鎖ホストでブロック」。
2. **その後の修正 (本セッション)**: 4 件の「短い修正」を全て適用し、付随して 1 件 (chebi20) が無料で復活。**11/18 が動作可能、7/18 が credential / 外部ホスト依存**。

## サマリ表 (更新後)

| 評価器 | 状態 | 備考 |
|--------|------|------|
| clinvar | ✅ 足固め済 | 第1波 |
| gnomad_af_correlation | ✅ 足固め済 | 第1波 |
| proteingym | ✅ 足固め済 | 第1波 (Zenodo 移行済み) |
| moleculenet | ✅ 足固め済 | 第1波 |
| moses | ✅ 足固め済 | 第1波 |
| protein_foldability | ✅ 足固め済 | 第1波 |
| chembl_scaffold_heldout | ✅ 足固め済 | 第1波の最後 |
| **rna_benchmark** | ✅ **足固め済** (本セッション) | parquet → JSONL prep + 直接 token-id 経路 + per-cell CI |
| **molecule_nat_lang** | ✅ **足固め済** (本セッション) | Mol-Instructions HF → pair CSV prep + 長さ stratified |
| **chemllmbench** | ✅ **足固め済 (7/9 sub-task カバー)** | 当初 downloader URL 修正 + 3 subtask の prep_jsonl + 予測ログ。本セッションでさらに 4 subtask を追加: `name_conversion` (`name_prediction/llm_test.csv` で SMILES→IUPAC、600 行)、`retrosynthesis` (`retro/uspto50k_retro_test.csv`、100 行)、`yield_prediction` (`yield_prediction/BH_sample_100_test.npz`、100 行 — TASK_TYPE を regression→exact に修正、上流データが Yes/No バイナリだったため)、`property_prediction` (5 csv ユニオン: BBBP/BACE/ClinTox/HIV/Tox の `_test.csv`、計 498 行、metadata に `dataset` を保持)。残 2 subtask (`text_guided_generation` / `smiles_understanding`) は上流に対応物が見当たらず未対応 |
| **chebi20** | ✅ **動作確認** (本セッション) | temperature=0 NaN は adapter 側 bug。修正で復活 |
| **deeploc** | ✅ **足固め済** (本セッション) | DTU の click-through landing は直 URL があり、auth 不要だった。downloader 自動化 + multi-label → single-label reshape + class-balanced subsample + cluster split + bootstrap CI + per-kingdom 予測ログ |
| **gue** | ✅ **足固め済** (本セッション) | 公開ミラー `leannmlindsey/GUE` (gated=False、token 不要) を発見。`huggingface_hub.snapshot_download` で取得し、ミラー命名 (emp_H3, virus_covid, human_tf_*) を canonical 28-task 名に rename。class-balanced subsample + bootstrap CI + per-class confusion + 予測ログ |
| **tape** | ✅ **足固め済 (6/6 sub-task カバー)** | songlabdata S3 は 403 のままだが、`AI4Protein/TAPE_Fluorescence`、`AI4Protein/TAPE_Stability`、`proteinea/remote_homology` の 3 つの公開ミラーで 3 sub-task を復活。さらに `proteinea/secondary_structure_prediction` で SS3/SS8 (per-residue) を追加対応 — adapter API に `embed_per_residue` を新設し、CB513 train/test を Q3/Q8/F1_macro + protein-level bootstrap CI で評価。本セッションで `proteinglm/contact_prediction_binary` (test 1505 / train 12041 protein、`label` は contact pair の `[i,j]` 配列) を発見し contact_prediction も復活: 各 train protein から K positives + K negatives を long-range サンプリング、residue embedding の elementwise product を pair feature にして LogReg head 学習、test では全 long-range pair (default `\|i-j\|≥24`) を score して precision@L/k (k=1,2,5) + per-protein bootstrap CI を計算。`predictions.jsonl` に各 protein の top-L/5 pair を score+label 込みで保存 |
| **replogle_perturb_seq** | ✅ **足固め済** (本セッション) | CellxGene H5AD は 403 のままだが、TruthSeq の figshare release (10.6084/m9.figshare.31840141) が同じ Replogle 2022 K562 atlas を 154 MB の long-format parquet で公開。pivot で wide 化、Ensembl REST で HGNC→ENSG 解決 (rna BERT vocab に必要)、delta-strength stratified subsample + bootstrap CI (perturbation 単位のリサンプリング) + best/worst-fit predictions log |
| **tabula_sapiens** | ✅ **足固め済** (本セッション) | 旧デフォルト URL は 403 だが、Tabula Sapiens collection (e5f58829-...) の他の H5AD は token なしで HEAD 200。Testis slice (0.39 GB) を default にし、prepare_jsonl で H5AD → top-N gene-id JSONL を生成。adapter.embed が pre-tokenized int を受けるよう拡張、class-balanced subsample + bootstrap CI + per-class CORRECT/WRONG narrative |
| **cosmic** | ✅ **足固め済** (本セッション) | Sanger の SPA を解析して NextAuth credentials login → `/api/mono/products/v1/downloads/download-file` 経由の presigned-URL ダウンロードを確立。CMC v100 (Cancer Mutation Census, alldata-cmc) 273 MB tar を sha256 検証付きで取得。`prepare_csv.py` が CMC TSV → 旧スキーマ CSV 変換 (Ensembl GRCh37 REST で ±flank bp の reference/variant sequence を構築、`MUTATION_SIGNIFICANCE_TIER` を `FATHMM_PREDICTION`/`DAMAGING|NEUTRAL` にマッピング)。さらに足固め化: `sample_cosmic` で tier 別 stratified サンプリング、`bootstrap_binary_ci` で auroc/auprc/accuracy/f1/sens/spec の 95% CI、`predictions_log.py` で per-row JSONL + 4 quadrant narrative TXT を出力。100 例 stratified smoke で 4-column CI table が REPORT.md に出力されることを確認 |
| omim | 🔒 credential 申請中 | OMIM_API_KEY (申請から ~1 営業日で承認) |

## 鍵となった発見と修正

### temperature=0 が NaN を引き起こす adapter bug
- `molcrawl/gpt2/model.py` の `generate()` は `logits / temperature` を計算するため、
  `temperature == 0` で inf → softmax で NaN → multinomial で
  `RuntimeError: probability tensor contains either inf, nan or element < 0` を吐く。
- 「molecule_nat_lang ckpt の重み病理」と切り分けていた問題は、実態は
  この adapter 側の数値安定化漏れだった。
- 修正: `_adapters/gpt2_adapter.py` で `eff_temp = max(temperature, 1e-5)` を強制。
  greedy decoding は softmax を peaky にするので機能上は同等。
- 副作用: chebi20, chemllmbench, 将来 generation を使う任意の評価器が同時に解放。

### HfMlm adapter が token-id 直接入力を受けるよう拡張
- rna_benchmark は parquet に既に int16 token-id を持っている。従来の
  `tokens_to_strings` で文字列化してから adapter で再 tokenize する経路は、
  rna BERT の WordLevel tokenizer が `[UNK]` を vocab に持たない罠で
  破綻していた。
- 修正: `score_likelihood(inputs)` の各要素が `list[int]` の場合、tokenizer を
  バイパスして直接 PLL を計算。文字列入力との互換性は維持。

### chemllmbench downloader URL の不一致
- 上流レポは `data/<task>/<file>.csv` の 2 階層。downloader は
  `data/<task>.jsonl` を取りに行って全 404。
- `prepare_jsonl.py` を新設して 3 つの flat-CSV subtask
  (molecule_captioning / molecule_design / reaction_prediction) を JSONL 化。
- 残 6 subtask は外部 Box / GitHub の追加 download が必要。

## 数値変化

- 当初: 7/18 working (39 %)
- セッション 1 後: 11/18 working (61 %) — rna_benchmark / molecule_nat_lang /
  chemllmbench / chebi20 を解放
- セッション 2 後: 12/18 working (67 %) — deeploc を解放
- セッション 3 後: 13/18 working (72 %) — gue を解放
- セッション 4 後: 14/18 working (78 %) — tape を解放 (3 sub-task)
- セッション 5 後: 15/18 working (83 %) — replogle_perturb_seq を解放
- セッション 6 後: 16/18 working (89 %) — tabula_sapiens を解放
- セッション 7 後: **17/18 working (94 %)** — cosmic を解放 (Sanger SPA reverse-engineering + Ensembl REST flank 構築)
- 残 1 は OMIM API キー承認待ち

cosmic 解放の鍵: Sanger は v98 前後で `/cosmic/file_download/...` を SPA-only
flow に閉鎖したが、SPA (Next.js + NextAuth.js) を解析すると 3 段階の
スクリプタブルなフローが存在することが判明 — (1) `/api/auth/csrf` →
`/api/auth/callback/credentials` で session cookie を取得、(2)
`/cosmic/download/cosmic` の RSC streaming payload (`__next_f.push(...)`) に
全プロダクトのカタログ (project_code / release_version / s3_object / sha256)
が server-render される、(3)
`/api/mono/products/v1/downloads/download-file?path=<s3_object>&bucket=downloads`
が presigned S3 URL を返す。`eval-data-cosmic.sh` がこの 3 段を駆動。
v100 では FATHMM_PREDICTION 列が `Cancer Mutation Census (alldata-cmc)` 製品の
`MUTATION_SIGNIFICANCE_TIER` (1/2 = driver, 3 = passenger) に置換されており、
`prepare_csv.py` がこの列を旧スキーマの DAMAGING/NEUTRAL に正規化しつつ
Ensembl GRCh37 REST API (`/sequence/region/...`) で ±flank bp の reference
sequence を取り、中心塩基を ALT で置換して variant_sequence を構築する
(15-55k req/hr の rate-limit があり 200-500 例なら数十秒で完走)。

tape 解放の鍵: 公式 songlabdata S3 は 403 のままだが、Hugging Face 上に
3 つの独立ミラー (`AI4Protein/TAPE_Fluorescence`、`AI4Protein/TAPE_Stability`、
`proteinea/remote_homology`) が gated=False で公開されていた。
それぞれスキーマが異なる (列名 `aa_seq`/`label` vs `primary`/`fold_label`)
ので、prepare_csv で TAPE canonical schema (`primary` + task-specific label)
に正規化して JSONL を吐くアプローチ。secondary_structure_* と
contact_prediction は per-residue 対応が必要なため未対応。

## 短期で次に取れる打ち手

1. **CellxGene の代替 URL 探索** → replogle / tabula_sapiens 復活の可能性
2. **dashboard が `bootstrap_ci_95` を rendering** — 14 評価器が CI を吐くようになったので、
   docs-src 側で error bar を出せるようになる
3. **secondary_structure_* / contact_prediction を per-residue 対応** — TAPE 残 3 sub-task を埋めるなら必要
4. **cosmic** — Sanger 側 API 変更で MolCrawl からは自動 DL 不可。手動取得 + `ed_register_existing` で manifest 化する経路が確立済み。ユーザがブラウザで download portal にログインしてファイルを `$LSD/eval/cosmic/` に置けば再走で取り込まれる
5. **omim** — API キー承認待ち。届き次第 `OMIM_API_KEY` を `.env` に追加して再走

## .env 自動ロードに関する注意

- `workflows/common_functions.sh` がリポジトリ直下の `.env` を自動 source する
  (なければ `~/.config/molcrawl/.env`)。
- 当初は `set -a; . .env; set +a` を使っていたが、これは bash の word-splitting +
  変数展開を経由するため (a) パスワードに `$` が含まれる場合にユーザー側で
  escape を要求してしまい、(b) caller が既に `set -u` を有効化していた状態
  だと `unbound variable` で死ぬ。
- 修正版は `KEY=VALUE` 行を直接パース (両端の `"` / `'` のみ剥がす) して
  `export` するシンプルなパーサ。bytes-for-bytes の値保存と strict-mode
  耐性を両立する。
- `check_learning_source_dir` も `[ -z "${LEARNING_SOURCE_DIR:-}" ]` に
  修正して `set -u` 下で死なないようにした。
