#!/usr/bin/env bash
# Run the GPU-required steps on a GCP A100 40 GB spot VM.
#
# Days 13–14 of the build:
#   - Day 13: pre-compute the 3 demo case fixtures with real RFdiffusion + AF-Multimer
#   - Day 14: Unsloth QLoRA fine-tune of Gemma 4 E4B on OAS antibodies
#
# Cheapest path:
#   gcloud compute instances create idiotypeforge-a100-spot \
#     --zone us-central1-c \
#     --machine-type a2-highgpu-1g \
#     --provisioning-model SPOT \
#     --instance-termination-action STOP \
#     --image-family pytorch-latest-gpu --image-project deeplearning-platform-release \
#     --boot-disk-size 200GB --maintenance-policy TERMINATE
#
# Total expected GCP spend: ~$25 across both days.

set -euo pipefail
cd "$(dirname "$0")/.."

# Real GPU runs — turn mocks OFF.
export IDIOTYPEFORGE_USE_MOCKS=0

echo "→ [1/3] Pre-compute demo fixtures with real RFdiffusion + AF-Multimer…"
for case in fl_carlotti2009 cll_subset2 dlbcl_young2015; do
  echo "  · $case"
  uv run python -m app.agent.orchestrator \
    --vh "data/demo_cases/${case}.fasta" \
    --vl "data/demo_cases/${case}.fasta" \
    --hla "HLA-A*02:01,HLA-B*07:02" \
    --patient-id "$case" \
    --out "data/demo_cases/${case}/"
done

echo "→ [2/3] Fine-tune Gemma 4 E4B with Unsloth QLoRA…"
uv run python scripts/finetune_gemma4_unsloth.py \
  --train-jsonl data/oas/train_200k_instruct.jsonl \
  --eval-jsonl  data/oas/eval_5k.jsonl \
  --epochs 1 \
  --push-to-hub

echo "→ [3/3] Run benchmarks notebook for the writeup…"
uv run jupyter nbconvert --execute notebooks/02_benchmarks.ipynb

echo
echo "✓ GCP A100 phase complete. Push fixtures + adapter to repo / HF Hub."
