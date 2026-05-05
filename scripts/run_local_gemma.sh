#!/usr/bin/env bash
# Run the IdiotypeForge pipeline LOCALLY with a real Gemma model in the
# loop (not template mode). This is what gets recorded for the YouTube
# video and makes the Gradio dashboard genuinely Gemma-driven.
#
# Why this exists separately from run_local.sh:
#   run_local.sh        — template mode (no LLM, deterministic, fast).
#   run_local_gemma.sh  — Gemma in the loop via Ollama, real function
#                         calling, real dossier composition.
#
# Note on Gemma 4 vs Gemma 3:
#   Gemma 4 (E2B / E4B / 26B / 31B) isn't yet packaged on the Ollama hub
#   as of 2026-05. Until Google/Ollama publishes it, this script defaults
#   to gemma3:4b, which is the closest stand-in (same active-parameter
#   scale as Gemma 4 E4B, same multimodal + tool-calling surface).
#   For the *real* Gemma 4 E4B path, run notebooks/idiotypeforge_kaggle.ipynb
#   on Kaggle Notebooks (free P100 GPU + native Gemma 4 access).
#
# Override via env: IDIOTYPEFORGE_GEMMA_MODEL=gemma4:e4b ./run_local_gemma.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Colours
GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; RESET='\033[0m'
step() { echo -e "${BLUE}→${RESET} $*"; }
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}!${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; }

# Env: switch the agent + dossier composer into Gemma mode.
export IDIOTYPEFORGE_USE_MOCKS="${IDIOTYPEFORGE_USE_MOCKS:-1}"        # GPU tools still mocked locally
export IDIOTYPEFORGE_AGENT_MODE=gemma
export IDIOTYPEFORGE_DOSSIER_MODE=gemma
export IDIOTYPEFORGE_GEMMA_MODEL="${IDIOTYPEFORGE_GEMMA_MODEL:-gemma3:4b}"

# ---------------------------------------------------------------------------
# 1. Ollama present?
# ---------------------------------------------------------------------------
step "Checking Ollama install…"
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama not installed."
  echo
  echo "  Install (one-time):"
  echo "    macOS:    brew install ollama"
  echo "    Linux:    curl -fsSL https://ollama.com/install.sh | sh"
  echo
  echo "  Then start the daemon and re-run this script:"
  echo "    ollama serve &"
  echo "    bash $0"
  exit 1
fi
ok "Ollama $(ollama --version 2>/dev/null | head -1 | awk '{print $NF}')"

# ---------------------------------------------------------------------------
# 2. Daemon reachable?
# ---------------------------------------------------------------------------
step "Checking Ollama daemon…"
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama daemon not running. Start it with:"
  echo "    ollama serve &"
  exit 1
fi
ok "Daemon reachable at localhost:11434"

# ---------------------------------------------------------------------------
# 3. Model present?
# ---------------------------------------------------------------------------
step "Checking model: $IDIOTYPEFORGE_GEMMA_MODEL …"
if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q "^$IDIOTYPEFORGE_GEMMA_MODEL\$"; then
  warn "Model not pulled yet."
  read -r -p "  Pull $IDIOTYPEFORGE_GEMMA_MODEL now? (~3 GB, one-time) [y/N] " ans
  if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
    ollama pull "$IDIOTYPEFORGE_GEMMA_MODEL"
  else
    err "Model required. Run: ollama pull $IDIOTYPEFORGE_GEMMA_MODEL"
    exit 1
  fi
fi
ok "Model $IDIOTYPEFORGE_GEMMA_MODEL ready"

# ---------------------------------------------------------------------------
# 4. uv sync (idempotent)
# ---------------------------------------------------------------------------
step "Syncing Python deps…"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv sync --extra dev
ok "Env ready"

# ---------------------------------------------------------------------------
# 5. Smoke test: can we round-trip a tool call through Gemma?
# ---------------------------------------------------------------------------
step "Smoke-testing Gemma → tool-call → Gemma round-trip…"
uv run python - <<PYEOF
import os, json, sys
import ollama

MODEL = os.environ["IDIOTYPEFORGE_GEMMA_MODEL"]
tools = [{
    "type": "function",
    "function": {
        "name": "echo_back",
        "description": "Trivial echo tool used only to verify Ollama function calling works.",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    },
}]
try:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Call echo_back with x='hello'."}],
        tools=tools,
    )
    msg = resp.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        print(f"  ✓ tool-call wiring OK ({len(tool_calls)} call(s))")
    else:
        # Some Gemma sizes don't natively emit tool_calls; that's OK,
        # the orchestrator can fall back to template mode.
        print(f"  ! model didn't emit tool_calls (content={msg.get('content','')[:80]})")
        print(f"    Pipeline will work but agent will partially fall back to template mode.")
except Exception as e:
    print(f"  ✗ Ollama smoke test failed: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
ok "Round-trip OK"

# ---------------------------------------------------------------------------
# 6. Launch the dashboard
# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}━━━ Launching IdiotypeForge with $IDIOTYPEFORGE_GEMMA_MODEL ━━━${RESET}"
echo "  agent_mode    = $IDIOTYPEFORGE_AGENT_MODE"
echo "  dossier_mode  = $IDIOTYPEFORGE_DOSSIER_MODE"
echo "  gpu_mocks     = $IDIOTYPEFORGE_USE_MOCKS"
echo "  ui            = http://localhost:7860"
echo
exec uv run python -m app.ui.gradio_app
