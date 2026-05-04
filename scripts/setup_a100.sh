#!/usr/bin/env bash
# IdiotypeForge — A100 spot VM setup script.
#
# Run this ONCE after SSH-ing into the freshly-provisioned GCP A100 spot VM
# (Deep Learning VM image with PyTorch + CUDA preinstalled). Idempotent —
# safe to re-run if a step failed mid-way.
#
# What it installs:
#   - System packages: tmux, mmseqs2, ncbi-blast+, build tools
#   - uv (Astral package manager)
#   - The IdiotypeForge repo + its `gpu` and `igfold` extras
#   - RFdiffusion (cloned to ~/RFdiffusion, weights downloaded)
#   - ProteinMPNN (cloned to ~/ProteinMPNN, weights bundled)
#   - ColabFold (for AlphaFold-Multimer rescoring)
#   - Unsloth + bitsandbytes + peft + trl
#
# After this completes, set the env vars in scripts/run_gcp_a100.sh and
# launch the workload.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; RESET='\033[0m'
step() { echo -e "${BLUE}→${RESET} $*"; }
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}!${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# 0. CUDA sanity check
# ---------------------------------------------------------------------------
step "Verifying GPU…"
if command -v nvidia-smi >/dev/null 2>&1; then
  # `nvidia-smi -L` is short-output; piping `nvidia-smi | head -3` triggers
  # SIGPIPE under `set -euo pipefail` and kills the script before driver init.
  nvidia-smi -L || true
else
  err "nvidia-smi missing. Use the deeplearning-platform-release pytorch-latest-gpu image."
  exit 1
fi
ok "GPU detected"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
step "Installing system packages…"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    tmux git curl wget build-essential \
    mmseqs2 ncbi-blast+ \
    hmmer
ok "System packages installed"

# ---------------------------------------------------------------------------
# 2. uv + Python deps
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  step "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | awk '{print $2}')"

step "Cloning idiotypeforge…"
if [ ! -d "$HOME/idiotypeforge" ]; then
  git clone https://github.com/aniruddhgoteti/idiotypeforge.git "$HOME/idiotypeforge"
fi
cd "$HOME/idiotypeforge"
git pull --quiet || true

step "Syncing Python env (base + gpu extra)…"
# NOTE: `--extra igfold` is intentionally omitted here. ablang2>=0.3.0 has
# no Python 3.13 wheel, and the Deep Learning VM ships 3.13 by default; the
# GPU workload (fixtures, finetune) only needs the `gpu` extra. IgFold /
# AbLang2 are CPU-side UI features.
uv sync --extra gpu
source .venv/bin/activate
ok "Env: $(python --version)"

# ---------------------------------------------------------------------------
# 3. RFdiffusion
# ---------------------------------------------------------------------------
step "Setting up RFdiffusion…"
if [ ! -d "$HOME/RFdiffusion" ]; then
  git clone https://github.com/RosettaCommons/RFdiffusion.git "$HOME/RFdiffusion"
fi
cd "$HOME/RFdiffusion"
mkdir -p models
cd models

# Download model weights if missing (from Baker Lab IPD CDN)
declare -A RFD_WEIGHTS=(
  ["Base_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt"
  ["Complex_base_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
  ["Complex_beta_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt"
  ["Complex_Fold_base_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5d7adbb5f4dc/Complex_Fold_base_ckpt.pt"
)
for fname in "${!RFD_WEIGHTS[@]}"; do
  if [ ! -f "$fname" ]; then
    step "  · downloading $fname"
    wget -q "${RFD_WEIGHTS[$fname]}"
  fi
done
ok "RFdiffusion weights ready"

cd "$HOME/RFdiffusion"
step "Installing RFdiffusion deps + SE3Transformer…"
pip install -q -e env/SE3Transformer || warn "SE3Transformer install failed (sometimes flaky); retry if RFdiffusion crashes"
pip install -q hydra-core omegaconf icecream pyrsistent

ok "RFdiffusion ready at $HOME/RFdiffusion"

# ---------------------------------------------------------------------------
# 4. ProteinMPNN
# ---------------------------------------------------------------------------
step "Setting up ProteinMPNN…"
if [ ! -d "$HOME/ProteinMPNN" ]; then
  git clone https://github.com/dauparas/ProteinMPNN.git "$HOME/ProteinMPNN"
fi
ok "ProteinMPNN ready at $HOME/ProteinMPNN"

# ---------------------------------------------------------------------------
# 5. ColabFold (AlphaFold-Multimer)
# ---------------------------------------------------------------------------
step "Installing ColabFold (this pulls JAX + AlphaFold deps; ~2 min)…"
cd "$HOME/idiotypeforge"
pip install -q "colabfold[alphafold]" --extra-index-url https://pypi.nvidia.com
ok "colabfold_batch on PATH"

# ---------------------------------------------------------------------------
# 6. Unsloth + fine-tuning stack
# ---------------------------------------------------------------------------
step "Installing Unsloth + bitsandbytes + peft + trl…"
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q bitsandbytes peft trl datasets accelerate
ok "Unsloth ready"

# ---------------------------------------------------------------------------
# 7. Env file for run_gcp_a100.sh
# ---------------------------------------------------------------------------
step "Writing env file for the runner…"
cat > "$HOME/idiotypeforge/.env.a100" <<EOF
# Source this file before running scripts/run_gcp_a100.sh
export RFDIFFUSION_DIR=\$HOME/RFdiffusion
export PROTEINMPNN_DIR=\$HOME/ProteinMPNN
export IDIOTYPEFORGE_USE_MOCKS=0
# Set HF_TOKEN below to push the LoRA adapter:
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
ok ".env.a100 written"

# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}━━━ A100 setup complete ━━━${RESET}"
echo
echo "Next:"
echo
echo "  source ~/idiotypeforge/.env.a100"
echo "  cd ~/idiotypeforge"
echo
echo "  # Day 13: pre-compute real RFdiffusion + AF-Multimer fixtures"
echo "  bash scripts/run_gcp_a100.sh fixtures"
echo
echo "  # Day 14: Unsloth fine-tune of Gemma 4 E4B (5 hrs)"
echo "  bash scripts/run_gcp_a100.sh finetune"
echo
echo "  # Both in one go"
echo "  bash scripts/run_gcp_a100.sh all"
echo
