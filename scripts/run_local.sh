#!/usr/bin/env bash
# Run the full IdiotypeForge pipeline locally on CPU using Gemma 4 via Ollama.
#
# Prereqs:
#   - Ollama installed and running (https://ollama.com)
#   - `ollama pull gemma3:4b` (small Gemma 4 variant, ~5 GB)
#   - `uv sync` to install Python deps
#
# This script runs the methodology validation path with mocks for the two
# GPU-only tools. Ship-quality output from real GPU runs is in `data/demo_cases/<id>/`.

set -euo pipefail

cd "$(dirname "$0")/.."

# Mocks ON for local; flip to 0 only on Day 13+ on GCP A100.
export IDIOTYPEFORGE_USE_MOCKS="${IDIOTYPEFORGE_USE_MOCKS:-1}"
export IDIOTYPEFORGE_GEMMA_MODEL="${IDIOTYPEFORGE_GEMMA_MODEL:-gemma3:4b}"

echo "→ Verifying Ollama is reachable…"
if ! curl -fsS http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "✗ Ollama not running. Start with: \`ollama serve\`"
  exit 1
fi

echo "→ Verifying Gemma 4 model is pulled…"
if ! ollama list | grep -q "gemma3:4b"; then
  echo "→ Pulling gemma3:4b (one-time, ~5 GB)…"
  ollama pull gemma3:4b
fi

echo "→ Running pipeline on FL demo case…"
uv run python -m app.agent.orchestrator \
  --vh data/demo_cases/fl_carlotti2009.fasta \
  --vl data/demo_cases/fl_carlotti2009.fasta \
  --hla "HLA-A*02:01,HLA-B*07:02" \
  --patient-id fl_001 \
  --out runs/

echo
echo "✓ Done. See runs/fl_001/events.jsonl for the agent trace."
echo "  To launch the UI: uv run python -m app.ui.gradio_app"
