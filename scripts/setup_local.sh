#!/usr/bin/env bash
# IdiotypeForge — one-shot local setup.
#
# Installs uv if missing, syncs the Python environment, runs unit tests,
# and prints next-step commands. Designed to be safe to re-run.
#
# What it does:
#   1. Checks Python 3.11+
#   2. Installs `uv` (Astral's fast Python package manager) if missing
#   3. Runs `uv sync` to create .venv and install all base + dev deps
#   4. Verifies Ollama is installed (warns if missing)
#   5. Runs the unit tests against the mock-mode pipeline
#   6. Prints next-step commands

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
RESET='\033[0m'

step() { echo -e "${BLUE}→${RESET} $*"; }
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}!${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Python check
# ---------------------------------------------------------------------------
step "Checking Python ≥ 3.11…"
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found. Install from https://python.org or via Homebrew."
  exit 1
fi
PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  err "Python $PY_VERSION found, need ≥ 3.11."
  exit 1
fi
ok "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# 2. uv installer
# ---------------------------------------------------------------------------
step "Checking for uv…"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found. Installing from https://astral.sh/uv …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installer puts uv into ~/.local/bin or ~/.cargo/bin depending on the OS
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    err "uv installed but not on PATH. Add ~/.local/bin to your PATH and re-run."
    exit 1
  fi
fi
ok "uv $(uv --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 3. Sync env
# ---------------------------------------------------------------------------
step "Creating virtualenv and installing base + dev deps (uv sync)…"
uv sync --extra dev
ok "Environment ready at $ROOT/.venv"

# ---------------------------------------------------------------------------
# 4. Ollama check (informational only — not required for tests)
# ---------------------------------------------------------------------------
step "Checking for Ollama (needed to run the agent locally)…"
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama not installed. Tests will still pass, but the live agent needs it."
  warn "Install from https://ollama.com (one-line install for macOS / Linux)."
else
  ok "Ollama $(ollama --version 2>/dev/null | head -1)"
  if ! ollama list 2>/dev/null | grep -q "gemma3:4b"; then
    warn "Gemma 4 model not pulled yet. When ready: \`ollama pull gemma3:4b\` (~5 GB)."
  else
    ok "gemma3:4b is available"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Run unit tests (mocks ON, no GPU required)
# ---------------------------------------------------------------------------
step "Running unit tests (mock-mode, CPU only)…"
export IDIOTYPEFORGE_USE_MOCKS=1
if uv run pytest tests/test_tools.py -v --tb=short; then
  ok "All unit tests passed"
else
  err "Some unit tests failed — see output above. Stub tests are expected to "
  err "return 'stub_not_implemented' until Days 2–8."
  # don't exit; this is expected during the scaffold phase
fi

# ---------------------------------------------------------------------------
# 6. Next steps
# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}━━━ Setup complete ━━━${RESET}"
echo
echo "Next steps:"
echo
echo "  • Launch the dashboard:"
echo "      uv run python -m app.ui.gradio_app"
echo "      → http://localhost:7860"
echo
echo "  • Run the pipeline on a demo case:"
echo "      uv run python -m app.agent.orchestrator \\"
echo "          --vh data/demo_cases/fl_carlotti2009.fasta \\"
echo "          --vl data/demo_cases/fl_carlotti2009.fasta \\"
echo "          --hla 'HLA-A*02:01' --patient-id fl_001 --out runs/"
echo
echo "  • Push this repo to your GitHub:"
echo "      see docs/PUSH_TO_GITHUB.md"
echo
echo "  • Run the full verification harness later:"
echo "      uv run pytest tests/ -v"
echo
