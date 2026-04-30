#!/usr/bin/env bash
# IdiotypeForge — A100 spot VM runner.
#
# Run on the GCP A100 VM after `bash scripts/setup_a100.sh` has completed.
#
# Subcommands:
#   fixtures   — pre-compute real RFdiffusion + AlphaFold-Multimer outputs
#                for the 3 demo cases (FL, CLL, DLBCL). Saves to
#                data/demo_cases/<case>/fixtures/
#   finetune   — Unsloth QLoRA fine-tune of Gemma 4 E4B on OAS antibodies.
#                Saves to runs/gemma4-e4b-ab-lora/. If HF_TOKEN is set,
#                pushes to huggingface.co/$HUB_ID
#   all        — run fixtures then finetune
#
# Required env (sourced from .env.a100):
#   RFDIFFUSION_DIR         path to RFdiffusion repo
#   PROTEINMPNN_DIR         path to ProteinMPNN repo
#   IDIOTYPEFORGE_USE_MOCKS=0   force real GPU runs
# Optional:
#   HF_TOKEN                Hugging Face token for `--push-to-hub`
#   HUB_ID                  e.g. yourname/idiotypeforge-gemma4-e4b-ab-lora

set -euo pipefail

cd "$(dirname "$0")/.."
SUBCMD="${1:-all}"

# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------
if [ -z "${IDIOTYPEFORGE_USE_MOCKS:-}" ] || [ "${IDIOTYPEFORGE_USE_MOCKS}" != "0" ]; then
  echo "✗ IDIOTYPEFORGE_USE_MOCKS must be 0 for real GPU runs."
  echo "  Run:  source .env.a100"
  exit 1
fi
if [ -z "${RFDIFFUSION_DIR:-}" ] || [ ! -d "$RFDIFFUSION_DIR" ]; then
  echo "✗ RFDIFFUSION_DIR ($RFDIFFUSION_DIR) is unset or missing."
  exit 1
fi
if ! command -v colabfold_batch >/dev/null 2>&1; then
  echo "✗ colabfold_batch not on PATH. Did setup_a100.sh complete?"
  exit 1
fi
if [ ! -d ".venv" ]; then
  echo "✗ .venv missing. Re-run scripts/setup_a100.sh."
  exit 1
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# GPU monitoring — start an nvidia-smi sampling loop in the background that
# writes a CSV every 5 s for the duration of the workload. Used to verify we
# actually hit ~85-95% utilisation; auditable after the fact.
# ---------------------------------------------------------------------------
GPU_LOG_DIR="runs/gpu_monitoring"
mkdir -p "$GPU_LOG_DIR"
GPU_LOG="${GPU_LOG_DIR}/$(date +%Y%m%d-%H%M%S)-nvsmi.csv"

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "! nvidia-smi not found; skipping GPU monitoring."
    return
  fi
  echo "→ Starting nvidia-smi monitor → $GPU_LOG (5 s sampling)"
  nvidia-smi \
      --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv \
      -l 5 \
      > "$GPU_LOG" 2>&1 &
  GPU_MON_PID=$!
  trap 'kill $GPU_MON_PID 2>/dev/null || true' EXIT
}

print_gpu_summary() {
  if [ ! -f "$GPU_LOG" ] || [ ! -s "$GPU_LOG" ]; then
    return
  fi
  python3 - "$GPU_LOG" <<'PYSUM'
import csv, sys, statistics
path = sys.argv[1]
util_vals = []
with open(path) as fh:
    rdr = csv.reader(fh)
    next(rdr, None)  # header
    for row in rdr:
        if len(row) < 3:
            continue
        try:
            util = float(row[2].strip().rstrip("%"))
            util_vals.append(util)
        except ValueError:
            pass
if not util_vals:
    print("  (no samples)")
    sys.exit(0)
n = len(util_vals)
avg = statistics.mean(util_vals)
p50 = statistics.median(util_vals)
above_80 = sum(1 for v in util_vals if v >= 80) / n * 100
print(f"  samples:     {n} (~{n*5/60:.1f} min wallclock)")
print(f"  GPU util:    avg={avg:.1f}%  p50={p50:.1f}%  ≥80%: {above_80:.1f}% of time")
PYSUM
}

start_gpu_monitor

# ---------------------------------------------------------------------------
# Subcommand: fixtures (Day 13)
# ---------------------------------------------------------------------------
run_fixtures() {
  echo "━━━ Day 13 · Real fixture pre-compute ━━━"
  for case in fl_carlotti2009 cll_subset2 dlbcl_young2015; do
    out_dir="data/demo_cases/${case}/fixtures"
    if [ -f "${out_dir}/dossier.json" ]; then
      echo "→ ${case}: fixtures exist, skipping (delete to force rerun)"
      continue
    fi
    echo
    echo "→ Running ${case}…"
    mkdir -p "$out_dir"
    python -m app.agent.orchestrator \
        --vh "data/demo_cases/${case}.fasta" \
        --vl "data/demo_cases/${case}.fasta" \
        --hla "HLA-A*02:01,HLA-B*07:02" \
        --patient-id "${case}" \
        --out "${out_dir}/.." \
        --mode template
    # Move dossier + audit + events into the fixtures dir for shipping.
    mv "${out_dir}/../${case}/dossier.md" "${out_dir}/dossier.md" 2>/dev/null || true
    mv "${out_dir}/../${case}/audit.md"   "${out_dir}/audit.md"   2>/dev/null || true
    mv "${out_dir}/../${case}/events.jsonl" "${out_dir}/events.jsonl" 2>/dev/null || true
    echo "✓ ${case} fixtures → ${out_dir}/"
  done
  echo
  echo "✓ All 3 demo fixtures pre-computed. Commit:"
  echo "    git add data/demo_cases/*/fixtures/ && git commit -m 'Pre-compute real demo fixtures (A100)'"
}

# ---------------------------------------------------------------------------
# Subcommand: finetune (Day 14)
# ---------------------------------------------------------------------------
run_finetune() {
  echo "━━━ Day 14 · Unsloth QLoRA fine-tune of Gemma 4 E4B ━━━"

  # Ensure we have OAS data for training
  if [ ! -f "data/oas/oas_paired.fasta" ]; then
    echo "→ OAS not yet downloaded; running scripts/download_oas.py"
    python scripts/download_oas.py --out data/oas
  fi

  hub_args=""
  if [ -n "${HUB_ID:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
    hub_args="--push-to-hub --hub-id ${HUB_ID}"
    echo "→ Will push adapter to huggingface.co/${HUB_ID}"
  else
    echo "→ HF_TOKEN / HUB_ID not set; saving adapter locally only."
  fi

  python scripts/finetune_gemma4_unsloth.py \
      --base-model "unsloth/gemma-4-e4b" \
      --train-fasta "data/oas/oas_paired.fasta" \
      --out "runs/gemma4-e4b-ab-lora" \
      --epochs 1 --lr 2e-4 \
      --max-sequences 200000 \
      $hub_args

  echo
  echo "✓ Fine-tune complete. Adapter at runs/gemma4-e4b-ab-lora/lora/"
  if [ -n "${HUB_ID:-}" ]; then
    echo "  Public adapter: https://huggingface.co/${HUB_ID}"
  fi
  echo "  Eval metrics:   runs/gemma4-e4b-ab-lora/metrics.json"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$SUBCMD" in
  fixtures)  run_fixtures ;;
  finetune)  run_finetune ;;
  all)       run_fixtures; echo; run_finetune ;;
  *)
    echo "Usage: $0 {fixtures|finetune|all}"
    exit 1
    ;;
esac

echo
echo "━━━ Done ━━━"
echo
echo "→ GPU utilisation summary (from $GPU_LOG):"
print_gpu_summary
echo
echo "Push results back to git from your laptop:"
echo "    cd ~/idiotypeforge"
echo "    git pull && git add data/demo_cases runs/ && git commit -m 'GPU outputs'"
echo "    git push"
