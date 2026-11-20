#!/bin/bash
# Queue the 68 arms of the rna batch-size grid of 2026-11-20, one chain per arm.
#
# Each arm runs the same 309,657,600 blocks, so its wall-clock cost does not depend
# on its batch: steps fall as the batch rises and the step time rises with it. The
# time is the measured rate at global 2,560 times 120,960, which is why the link
# count comes from the probe's table and not from a guess. One spare link is added
# per arm so a lost segment does not end the arm.
#
# Usage:
#   ARMS=<arm inventory TSV> PROBE_TABLE=<probe table TSV> \
#     LEARNING_SOURCE_DIR=... MODEL_OUTPUT_ROOT=... SLURM_ACCOUNT=... \
#     [SUBMIT_RECORD=...] [DRY_RUN=1] ./workflows/submit-rna-batch-sweep.sh
#
# Env:
#   ARMS           the inventory from scripts/rna_batch_sweep_arms.py --out
#   PROBE_TABLE    the table from scripts/rna_microbatch_probe_table.py --out
#   LEARNING_SOURCE_DIR  the learning_source root holding rna/
#   MODEL_OUTPUT_ROOT    runs root, outside every input tree
#   SLURM_ACCOUNT  the account to charge
#   SUBMIT_RECORD  TSV of what was submitted; also routes through sbatch-watched.sh
#   ONLY_ARCH      bert or gpt2, to queue one framework
#   ONLY_SIZES     space-separated sizes, e.g. "small medium"
#   ONLY_BATCHES   space-separated batches, e.g. "256 4096"
#   SPARE_LINKS    extra links per arm beyond the measured need (default 1)
#   DRY_RUN        1 to print the plan and submit nothing
#
# The chain is afterany and bert-grid.sbatch decides for itself whether to continue:
# it resumes from a TIMEOUT and refuses a CANCELLED predecessor, so an outside
# cancellation stops the arm instead of restarting it somewhere unintended.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${ARMS:?set ARMS to the arm inventory TSV}"
: "${PROBE_TABLE:?set PROBE_TABLE to the probe table TSV the link counts come from}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside every input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
for f in "${ARMS}" "${PROBE_TABLE}"; do
  [ -f "${f}" ] || { echo "no such file: ${f}" >&2; exit 1; }
done
SPARE_LINKS="${SPARE_LINKS:-1}"
SEGMENT_HOURS=96          # the 4-day limit of this partition
BLOCKS_AT_2560=120960     # 309,657,600 blocks / 2,560: steps an arm of batch 2,560 runs

# Join the inventory to the measured rates and work out the links each arm needs.
PLAN="$(mktemp)"; trap 'rm -f "${PLAN}"' EXIT
awk -v FS='\t' -v OFS='\t' \
    -v only_arch="${ONLY_ARCH:-}" -v only_sizes="${ONLY_SIZES:-}" \
    -v only_batches="${ONLY_BATCHES:-}" -v spare="${SPARE_LINKS}" \
    -v seg_hours="${SEGMENT_HOURS}" -v ref_steps="${BLOCKS_AT_2560}" '
  function wanted(list, value) { return list == "" || index(" " list " ", " " value " ") }
  FILENAME == probe && /^#/ { next }
  FILENAME == probe && !probe_head { for (i = 1; i <= NF; i++) p[$i] = i; probe_head = 1; next }
  FILENAME == probe {
    if ($p["s_per_step_at_2560"] != "")
      rate[$p["arch"] " " $p["size"] " " $p["micro"]] = $p["s_per_step_at_2560"]
    next
  }
  /^#/ { next }
  !arm_head { for (i = 1; i <= NF; i++) a[$i] = i; arm_head = 1; next }
  {
    arch = $a["arch"]; size = $a["size"]; batch = $a["batch"]
    if (!wanted(only_arch, arch) || !wanted(only_sizes, size) || !wanted(only_batches, batch))
      next
    key = arch " " size " " $a["micro_batch"]
    if (!(key in rate)) { print "NO RATE", key > "/dev/stderr"; bad = 1; next }
    hours = ref_steps * rate[key] / 3600
    links = int(hours / seg_hours); if (hours > links * seg_hours) links++
    links += spare
    print arch, size, batch, $a["lr_slug"], $a["config"], links, \
          sprintf("%.0f", hours), $a["out_dir"]
  }
  END { if (bad) exit 3 }
' probe="${PROBE_TABLE}" "${PROBE_TABLE}" "${ARMS}" > "${PLAN}"

total_arms=$(wc -l < "${PLAN}")
total_links=$(awk -F'\t' '{s += $6} END {print s + 0}' "${PLAN}")
total_gpuh=$(awk -F'\t' '{s += $7 * 4} END {printf "%.0f", s}' "${PLAN}")
echo "=== rna batch-size grid: ${total_arms} arms, ${total_links} links, ~${total_gpuh} GPU-h"
echo "    commit $(git rev-parse --short HEAD), MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
[ "${total_arms}" -gt 0 ] || { echo "nothing to queue" >&2; exit 1; }

if [ -n "${SUBMIT_RECORD:-}" ] && [ -z "${DRY_RUN:-}" ]; then
  mkdir -p "$(dirname "${SUBMIT_RECORD}")"
  [ -s "${SUBMIT_RECORD}" ] || printf 'submitted_at\tarch\tsize\tbatch\tlr_slug\tlink\tjob_id\tdepends_on\tcommit\n' > "${SUBMIT_RECORD}"
fi

while IFS=$'\t' read -r arch size batch slug cfg links hours out; do
  # A real submission needs the config on disk. A dry run is most useful before the
  # configs are written -- it is how the plan gets read before anything is generated
  # -- so there a missing file is said out loud and the plan carries on.
  if [ ! -f "${cfg}" ]; then
    if [ -n "${DRY_RUN:-}" ]; then
      echo "    (not written yet: ${cfg})"
    else
      echo "missing config: ${cfg}" >&2; exit 1
    fi
  fi
  tag="${size}-b${batch}-lr${slug}"
  echo "  ${arch} ${tag}: ${links} link(s), ~${hours} h -> ${out}"
  dep=""
  for i in $(seq 1 "${links}"); do
    if [ "${arch}" = "bert" ]; then
      env="ALL,CONFIG=${cfg},SEGMENT=${i},MODALITY=rna"
      script="workflows/bert-grid.sbatch"
    else
      env="ALL,GRID_CONFIG=${cfg}"
      script="workflows/rna-gpt2-grid.sbatch"
    fi
    name="rna-${arch}-${tag}-L${i}"
    args=(--job-name="${name}" --account="${SLURM_ACCOUNT}" --export="${env}")
    [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")
    if [ -n "${DRY_RUN:-}" ]; then
      echo "    [dry-run] link ${i}${dep:+ after ${dep}}: sbatch ${args[*]} ${script}"
      dep="<${name}>"
      continue
    fi
    if [ -n "${SUBMIT_RECORD:-}" ]; then
      jid=$(workflows/sbatch-watched.sh "${name}" -- "${args[@]}" "${script}")
    else
      jid=$(sbatch --parsable "${args[@]}" "${script}")
    fi
    echo "    link ${i}: job ${jid}${dep:+ (after ${dep})}"
    [ -n "${SUBMIT_RECORD:-}" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -Is)" "${arch}" "${size}" "${batch}" "${slug}" "${i}" "${jid}" \
      "${dep}" "$(git rev-parse HEAD)" >> "${SUBMIT_RECORD}"
    dep="${jid}"
  done
done < "${PLAN}"
echo "=== ${total_arms} arms, ${total_links} links ==="
