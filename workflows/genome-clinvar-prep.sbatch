#!/bin/bash
#SBATCH --job-name=mc-gen-clinvar-prep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Build the ClinVar table the §7 evaluator reads.
#
# The evaluator wants reference_sequence / variant_sequence / ClinicalSignificance.
# Those are not in NCBI's variant_summary.txt.gz, which carries coordinates and
# no sequence, so the copy fetched by eval-data-clinvar.sh cannot be converted
# into them on its own. prepare_clinvar.py builds them instead: it pulls the
# variant list from HuggingFace and cuts a window out of the GRCh38 reference
# around each one, substituting the ALT allele for the variant copy.
#
# So this needs the network, and the cluster's DNS is broken on every node --
# hence the netrun wrapper, which bind-mounts a working resolv.conf in a user
# namespace. HF_HUB_OFFLINE is deliberately left unset here; it is set for
# training runs, and leaving it on would make the download fail rather than
# fall back.
#
# One GPU because the partition requires one, not because the work uses it.
#
# Output goes under a separate root from the training corpus, which is read-only
# to us: $CLINVAR_ROOT/genome_sequence/clinvar/. That is the layout
# molcrawl.core.paths expects (CLINVAR_DIR = GENOME_SEQUENCE_DIR + "/clinvar").

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# The interpreter lives in the main checkout: ./miniconda does not exist in a worktree.
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
[ -x "${PYROOT}/miniconda/bin/python" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

CLINVAR_ROOT="${CLINVAR_ROOT:-$PWD/learning_source_eval}"
NETRUN="${NETRUN:-$HOME/.molcrawl-prep/netrun.sh}"

[ -x "${NETRUN}" ] || { echo "no network wrapper at ${NETRUN}; DNS is broken without it"; exit 1; }

export LEARNING_SOURCE_DIR="${CLINVAR_ROOT}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PWD/tmp/hf-cache-genome"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "${CLINVAR_ROOT}/genome_sequence/clinvar"

echo "=== node=$(hostname) ==="
echo "=== output root: ${CLINVAR_ROOT}/genome_sequence/clinvar ==="
df -h "${CLINVAR_ROOT}" | tail -1 | sed 's/^/    /'

"${NETRUN}" env \
    LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}" \
    TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    PYTHONPATH="${PYTHONPATH}" \
    "${PYROOT}/miniconda/bin/python" -m molcrawl.data.genome_sequence.dataset.clinvar.prepare_clinvar
rc=$?

echo ""
echo "---- rc=${rc} ----"
echo "=== what was produced ==="
ls -la "${CLINVAR_ROOT}/genome_sequence/clinvar" 2>/dev/null | sed 's/^/    /'

# The evaluator only needs the CSV; the tokenised DatasetDict after it is for
# fine-tuning. Report on the CSV separately so a later failure in the tokenise
# step does not read as "no usable output".
CSV="${CLINVAR_ROOT}/genome_sequence/clinvar/clinvar_sequences.csv"
if [ -f "${CSV}" ]; then
    echo "=== clinvar_sequences.csv: $(du -h "${CSV}" | cut -f1), $(($(wc -l < "${CSV}") - 1)) rows ==="
    head -1 "${CSV}" | tr ',' '\n' | sed 's/^/    column: /'
else
    echo "=== clinvar_sequences.csv NOT produced ==="
fi
echo "=== done @ $(date -Is) ==="
