# ローカルRefSeq FASTAの1塩基前処理

ダウンロード済みのゲノムFASTAを、GPT-2用の1塩基トークンParquetに変換します。
トークナイザーの学習やダウンロードは行いません。

## ファイルの配置

| ファイル | 役割 |
|---|---|
| `molcrawl/preparation/genome/preprocess_fasta.py` | FASTA → raw → Parquetの前処理本体 |
| `molcrawl/preparation/genome/parallel.py` | 入力の探索・分割・プロセス並列実行 |
| `workflows/01-genome_sequence-prepare-refseq.slurm` | Slurmへの投入用 |
| `molcrawl/data/genome_sequence/dataset/refseq/preprocess_fasta.py` | 旧パスの互換用入口 |
| `tests/preparation/genome/test_parallel.py` | 探索・並列実行・互換性・実データ変換のテスト |

前処理本体は従来の `preprocess_fasta.py` から移動したもので、処理内容を変更していません。
Slurmファイルには個人の絶対パス、課金先、partitionを埋め込みません。

## 配列処理の条件

並列実行の入口はGPT-2用に以下の条件を明示します。

```text
--model gpt2 --chunk_size 1024 --n_split_threshold 10 --max_n_frac 0.1
```

- 元FASTA中のN/nが10個以上連続する場所で分割します。
- 大文字化し、ACGTN以外の文字をNへ変換します。
- 各raw行を1024塩基のチャンクに分け、端数を破棄します。
- チャンク内のN率が10%を超える場合、そのチャンクを除外します。
- `A=0, T=1, G=2, C=3, N=4` のIDへ変換します。

曖昧塩基をNに変換するのはN-run分割の後です。今回の配置変更ではこの順序も維持します。
短いNの並びは保持するため、Nが1個でも分割する
`fasta_to_raw_per_accession` の経路とは異なります。
出力は `input_ids` 列のみで、contigの出自やtrain/valid/test分割は含みません。

## 単一プロセスで実行

MolCrawlのPython環境を有効にし、リポジトリ直下で実行します。
`.fna.gz` などの圧縮FASTAも直接読めます。

```bash
python -m molcrawl.preparation.genome.preprocess_fasta \
  --input learning_source/refseq_genomes/fasta \
  --output learning_source/refseq_genomes/processed \
  --model gpt2 --n_split_threshold 10 --max_n_frac 0.1
```

## 並列実行の起動確認

必要なPythonパッケージはMolCrawl環境へインストールしてください。Parquet出力には `pyarrow` が必要です。
入力を探索する前に、以下で実際に読み込まれる前処理本体とCLIの互換性を確認できます。

```bash
python -m molcrawl.preparation.genome.parallel --check-molcrawl
```

## Slurmで32件の試験

リポジトリ直下で実行します。`ACCOUNT` と `PARTITION` は利用環境の値に置き換えます。

```bash
export MOLCRAWL_PYTHON="$(command -v python)"
sbatch -A ACCOUNT -p PARTITION \
  --cpus-per-task=2 --mem=256G --time=00:30:00 \
  workflows/01-genome_sequence-prepare-refseq.slurm pilot
```

最初の32件を見つけたら探索を止め、1プロセスで処理します。
探索に15秒以上かかる場合は進捗を表示します。全体の件数・サイズを調べるための全件探索はしません。
この32件は代表サンプルでも最大ゲノム試験でもないため、全件の時間や最大メモリを確定できません。

標準入力先は `$LEARNING_SOURCE_DIR/refseq_genomes/fasta/` です。
`LEARNING_SOURCE_DIR` が未設定なら、リポジトリ直下の `learning_source/` を使用します。
`REFSEQ_BASE` を設定すればRefSeqの基点を、`REFSEQ_INPUT` で入力だけを変更できます。

```bash
export REFSEQ_BASE=/path/to/refseq_genomes
```

試験出力は `$REFSEQ_BASE/processed_gpt2_n10_pilot/` です。
入力のFASTAはコピーせず、各グループの `input_links/` から参照します。
既知のNCBI CDS/RNAファイル名は除外しますが、内容による判定は行わないのでゲノムFASTAのみを指定してください。
入力ルートのシンボリックリンクは解決しますが、配下のディレクトリリンクはたどりません。

## 本番

```bash
export REFSEQ_WORKERS=8
sbatch -A ACCOUNT -p PARTITION \
  workflows/01-genome_sequence-prepare-refseq.slurm full
```

既定値は1ノード、32 CPU、メモリ1024GiB、8並列、実行時間の上限4日です。
これらは所要時間・メモリ消費の予測ではなく、既存の投入設定を引き継いだ初期値です。
必要に応じて `sbatch` のオプションで上書きしてください。
入力を全件探索して64組に分けるため、本番では全件探索の時間がかかります。
圧縮ファイルサイズを目安に分散しますが、実際の速度・メモリ消費とは比例するとは限りません。
前処理本体は1つのゲノムのトークンをまとめてメモリに保持するため、大型ゲノムは別にメモリ評価が必要です。

本番出力は `$REFSEQ_BASE/processed_gpt2_n10_parallel/` です。

```text
processed_gpt2_n10_parallel/
├── plan.json
├── summary.json
└── shard_000/ ... shard_063/
    ├── input_links/
    ├── preprocess.log
    ├── complete.json
    └── processed/
        ├── raw_files/
        └── parquet_gpt2/
```

## 進捗と終了確認

投入時のジョブ番号で確認します。

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 30 refseq-JOB_ID.log
```

`COMPLETED` と `ExitCode=0:0` が正常終了です。
実行結果は `summary.json`、個別の処理ログは `shard_*/preprocess.log` に保存します。
中断出力は元の前処理が安全に再開できないため、該当shardを出力先外へ退避してから再投入します。
`started.json` だけを消して再開しないでください。成功済みshardは再投入時にスキップします。
計画作成後は入力を変更しないでください。

## 旧パスと実行計画の互換性

旧コマンドも引き続き使用できます。

```bash
python -m molcrawl.data.genome_sequence.dataset.refseq.preprocess_fasta --help
```

旧パスからの名前付きimportも新しい本体へ転送します。
旧 `slurm_refseq/run_parallel.py` のN>=10版（plan version 4）と計画識別子・保存構成を維持しているため、
同じ入力・出力・分割数で実行する場合、完了済みshardを再利用できます。
Nが1個でも分割する版や、過去のplan version 2/3の計画は再利用しません。

## 検証

```bash
python tests/preparation/genome/test_parallel.py
```

`pyarrow` がある環境では実際のgzip FASTAからParquetまで確認します。
旧・新CLIの出力一致、短いNの保持、連続Nでの分割、N率フィルタ、並列実行、完了済みスキップを検証します。
`pyarrow` がない環境では実Parquetのテストのみスキップします。

## YAML-based entry point

Run preprocessing from the repository root:

    python -m molcrawl.preparation.preparation_script_genome_sequence assets/configs/genome_sequence.yaml

Inspect the command without starting preprocessing:

    python -m molcrawl.preparation.preparation_script_genome_sequence assets/configs/genome_sequence.yaml --dry-run

The YAML must set `data_preparation.pipeline: single_nucleotide`.
It controls `input_dir`, `output_dir`, `num_worker`, `num_shards`,
and `pilot_files`. Relative paths are resolved from the working directory.
This entry point does not accept the former BPE configuration fields.

Submit the configuration through Slurm:

    sbatch -A ACCOUNT -p PARTITION workflows/01-genome_sequence-prepare-refseq.slurm assets/configs/genome_sequence.yaml

In YAML mode, the configuration controls paths and parallelism.
REFSEQ_BASE and REFSEQ_WORKERS do not override those values.
Slurm resource requests must be supplied separately for the target cluster.
