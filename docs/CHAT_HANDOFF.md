# IdiotypeForge — Project Handoff

> Use this file to bootstrap a fresh chat session. Paste it into a new
> conversation alongside the question you want help with. Everything below
> reflects the state of the repo at commit `0e1f4cf`.

---

## 1. What this project is

**IdiotypeForge**: an AI agent that takes a tumour BCR sequence (VH + VL +
HLA) from a B-cell non-Hodgkin lymphoma patient and outputs a personalised
therapy dossier — three modalities (mRNA vaccine peptides, designed
bispecific scFv, autologous 4-1BBz CAR-T cassette), starting doses, off-
target safety, manufacturing brief — in under 2 hours instead of the
6-month hand-crafted process that killed the BiovaxID-era idiotype
vaccine trials.

**Built for the Kaggle Gemma 4 Good Hackathon.** Deadline: 18 May 2026.
Tracks: Main ($100K) + Health & Sciences ($10K) + Digital Equity ($10K) +
Unsloth fine-tuning ($10K) = $130K eligible.

**Repo**: <https://github.com/aniruddhgoteti/idiotypeforge> (public).
**Local path**: `/Users/aniruddhgoteti/workspace/idiotypeforge/`.
**Owner**: Aniruddh Goteti (`aniruddh.goteti@orbion.life`).

---

## 2. The personal frame (do not flatten this when responding)

The user is a **stage-IV Hodgkin lymphoma survivor** treated with
ABVD → DHAP × 2 cycles → 1 cycle pembro+GVD → autologous SCT. The project
is **not** about treating their (now-resolved) Hodgkin. It is for
patients with poor-prognosis subtypes (PTCL ~30 % 5-yr OS, ATLL <20 %,
primary CNS, R/R post-CAR-T, Burkitt in LMICs) who do not have the same
treatment menu. Equity, not novelty, is the soul of the project.

The README opens with this story in first person. Do not replace it with
US-centric statistics; the user explicitly asked for storytelling, not
statistics.

---

## 3. Architecture in one sentence

A Gemma 4 agent orchestrates **11 deterministic scientific tools** (10 +
a batched variant) via native function calling, with every numeric output
checked by **5 verification gates** including a ProvenanceGate that
catches hallucinated numbers. Pipeline runs on a laptop in template mode
(no LLM); GPU phase swaps in real RFdiffusion + AlphaFold-Multimer + a
fine-tuned Gemma 4 LoRA.

```
Patient BCR → Gemma 4 agent → 11 tools → 5 verification gates → therapy dossier
                                                                ├── 3 mRNA vaccine peptides
                                                                ├── 3 designed scFv binders
                                                                ├── CAR-T 4-1BBz cassette
                                                                ├── Patient-specific doses
                                                                └── Audit trail
```

The 11 tools (in `app/tools/`):

| Tool | What it does | Compute |
|---|---|---|
| `number_antibody` | ANARCI numbering (CDR1/2/3, V/J genes) | CPU |
| `predict_fv_structure` | IgFold structure prediction | CPU/GPU |
| `score_cdr_liabilities` | Regex scan for N-glyc, deamidation, isomerisation, oxidation, free Cys, fragmentation | CPU |
| `predict_mhc_epitopes` | MHCflurry 2.x HLA-I peptide prediction | CPU |
| `design_binder` | RFdiffusion + ProteinMPNN de novo binders | **GPU** (mock available) |
| `rescore_complex` | AlphaFold-Multimer per-candidate | **GPU** (mock available) |
| `rescore_complex_batch` | AlphaFold-Multimer batched, ColabFold loaded once | **GPU** |
| `offtarget_search` | MMseqs2 + BLAST vs OAS + UniProt | CPU (needs data download) |
| `assemble_car_construct` | 4-1BBz cassette template assembly | CPU |
| `render_structure` | Headless matplotlib 3D rendering | CPU |
| `estimate_doses` | Patient-specific starting doses (mRNA / bispecific / CAR-T), traced to published trials | CPU |
| `compose_dossier` | Final markdown dossier (template mode + Gemma mode stub) | CPU/GPU |

The 5 gates (in `app/verification/`):

1. **SchemaGate** — pydantic validation
2. **MockModeGate** — no `mock=True` outputs in production runs
3. **ThresholdGate** — hard numerical thresholds (pLDDT ≥ 0.80, ipLDDT ≥ 0.50, off-target identity < 70 %, etc.)
4. **ProvenanceGate** — every dossier number must trace to a tool output via the multi-format alias index
5. **CitationGate** — every `[Author Year]` must resolve to `data/references.bib`

---

## 4. Repo layout

```
idiotypeforge/
├── README.md                        # Personal opener + module walkthrough
├── LICENSE                          # CC-BY 4.0 (Kaggle Winner License)
├── pyproject.toml                   # uv-managed; gpu / igfold / dev / deploy extras
├── app/
│   ├── agent/
│   │   ├── orchestrator.py          # Template + Gemma modes; auto-fallback
│   │   ├── router.py                # 12 tool entries (10 tools + batch + dossier)
│   │   └── prompts/
│   │       ├── system.md            # Anti-hallucination rule baked in
│   │       └── dossier.md
│   ├── tools/                       # 10 deterministic tools + 2 mocks
│   ├── ui/
│   │   ├── gradio_app.py            # 4-tab dashboard, streaming agent log
│   │   ├── decision_card.py
│   │   └── saliency.py
│   ├── verification/
│   │   ├── gates.py                 # 5 gates + GateRunner
│   │   └── provenance.py            # ArtifactStore + numeric_aliases
│   └── calibration/
│       └── isotonic.py              # IsotonicRegression confidence wrapper
├── data/
│   ├── demo_cases/
│   │   ├── README.md                # Provenance documentation
│   │   ├── fl_carlotti2009.fasta    # IGHV4-34 + IGKV2-30 (representative FL)
│   │   ├── cll_subset2.fasta        # IGHV3-21 + IGLV3-21*R110D stereotyped
│   │   └── dlbcl_young2015.fasta    # IGHV3-23 + IGLV1-44 (representative GCB-DLBCL)
│   └── references.bib               # 19 entries; CitationGate is bound to this
├── scripts/
│   ├── setup_local.sh               # Install uv, sync env, run tests
│   ├── setup_a100.sh                # VM bootstrap (RFdiffusion + ColabFold + Unsloth)
│   ├── run_local.sh                 # Local Ollama-driven run
│   ├── run_gcp_a100.sh              # VM workload runner (fixtures | finetune | all)
│   ├── download_oas.py              # OAS healthy paired sample
│   ├── download_uniprot.py          # SwissProt human + BLAST DB
│   └── finetune_gemma4_unsloth.py   # QLoRA fine-tune of Gemma 4 E4B (L4-tuned defaults)
├── tests/                           # 110 pass, 7 intentionally skipped (heavy deps)
└── docs/
    ├── architecture.excalidraw      # Editable diagram source
    ├── EXPORT_DIAGRAM.md            # How to re-export the PNG
    ├── PUSH_TO_GITHUB.md            # 3 paths for pushing repo
    ├── A100_RUNBOOK.md              # *** TBD — see §6 ***
    ├── video_shotlist.md            # 3-minute YouTube shotlist
    ├── writeup.md                   # Kaggle 1500-word writeup (skeleton)
    └── CHAT_HANDOFF.md              # ← this file
```

---

## 5. Where the project stands today

### What works end-to-end
- **CPU pipeline runs** in under a minute on a laptop. 11 tool calls per
  case, all 5 verification gates green, a complete dossier (with doses)
  drops out as markdown.
- **3 demo cases work** (FL, CLL, DLBCL), with subtype-signature CDR3
  motifs verified.
- **110 tests pass** (`python3 -m pytest tests/`); 7 intentionally
  skipped behind `@pytest.mark.skipif(not HAVE_X)` decorators (ANARCI,
  IgFold, MHCflurry — they run automatically when those deps are
  installed).
- **GPU optimisations in place** for when we DO get on a GPU:
  batched ColabFold rescore, batched ProteinMPNN, Unsloth tuned for L4
  (batch=8, packing, num_workers=4), `nvidia-smi` monitoring loop.

### What's NOT done yet
- **The A100/L4 spot run never completed.** We provisioned VMs three
  times in one session; L4 spot got preempted twice before SSH was even
  ready, then we tried A100 spot in us-central1-a but the user paused
  before SSH came up. **All VMs deleted, $0.20–$0.50 burned total.**
- Real RFdiffusion fixtures for the demo cases — currently the
  pipeline uses calibrated mocks (deterministic, realistic-shaped, but
  synthetic).
- Real AlphaFold-Multimer rescore — same: mocked.
- Unsloth fine-tune of Gemma 4 E4B on OAS antibodies — needed for the
  Unsloth track ($10K).
- Hugging Face Space deployment — Day 15.
- 3-minute YouTube video — shotlist exists at `docs/video_shotlist.md`,
  but not shot.
- Cover image — not made.
- Final writeup polish — `docs/writeup.md` is a skeleton.

### Plan progress (from `~/.claude/plans/okay-give-me-a-piped-quilt.md`)

| Day | Status |
|---|---|
| 1 — Scaffold | ✅ |
| 2 — ANARCI + IgFold + CDR liabilities | ✅ |
| 3 — Download scripts | ✅ implemented (not yet run on data) |
| 4 — MHCflurry | ✅ |
| 5 — AF-Multimer mock + CAR + dossier | ✅ |
| 6 — Real demo BCR sequences (IMGT-germline-derived) | ✅ |
| 7 — Orchestrator wiring (template + Gemma + auto-fallback) | ✅ |
| 8 — Streaming tool-call log + UI integration | ✅ |
| 9 — Multimodal renders → Gemma 4 image input | 🔜 |
| 10 — `verify_pipeline.py` end-to-end tests | ✅ |
| 11 — Gradio UI v1 | ✅ |
| 12 — Decision cards + AbLang2 saliency + reliability diagram | 🟡 cards done, saliency/reliability pending |
| 13 — GCP A100: real RFdiffusion + AF-Multimer fixtures | 🟡 **code ready, VM run aborted** |
| 14 — Unsloth fine-tune Gemma 4 E4B | 🟡 **code ready, VM run aborted** |
| 15 — HF Space deploy + benchmarks notebook | 🔜 |
| 16 — Video shoot + cover image | 🔜 |
| 17–19 — Buffer + submit | 🔜 |

---

## 6. Resuming the GPU phase later — exact commands

The setup + workload scripts are pushed and ready. From a fresh chat:

### A. Auth + provision

```bash
# 1. Re-auth gcloud (the project's tokens expire after a few hours)
gcloud auth login
gcloud config set project studied-airline-440212-n8

# 2. Provision an A100 spot VM. We tried L4 first but got preempted twice
#    in 10 min — L4 spot capacity is constrained today. A100 has 16-quota
#    in us-central1 and generally better availability. Try multiple zones.
for zone in us-central1-a us-central1-b us-central1-c us-central1-f; do
  echo "→ Trying A100 in $zone…"
  if gcloud compute instances create idiotypeforge-a100-spot \
      --project=studied-airline-440212-n8 \
      --zone=$zone \
      --machine-type=a2-highgpu-1g \
      --provisioning-model=SPOT \
      --instance-termination-action=DELETE \
      --maintenance-policy=TERMINATE \
      --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=200GB \
      --boot-disk-type=pd-balanced \
      --metadata="install-nvidia-driver=True,enable-oslogin=FALSE" \
      --scopes=cloud-platform \
      --no-shielded-secure-boot \
      --tags=idiotypeforge 2>&1 | grep -qE "RUNNING"; then
    echo "✓ Created in $zone"
    export ZONE=$zone
    break
  fi
done
```

### B. Wait for SSH + run setup + workload in tmux

```bash
# 3. Wait for SSH (NVIDIA driver install takes ~3 min)
until gcloud compute ssh idiotypeforge-a100-spot --zone=$ZONE \
       --command='echo ready' --quiet >/dev/null 2>&1; do
  echo "  waiting…"; sleep 15
done

# 4. SSH in, kick off setup + workload in one detached tmux session
gcloud compute ssh idiotypeforge-a100-spot --zone=$ZONE -- bash <<'EOF'
set -e
sudo apt-get install -y -qq tmux
[ ! -d "$HOME/idiotypeforge" ] && git clone --quiet https://github.com/aniruddhgoteti/idiotypeforge.git "$HOME/idiotypeforge"
cd "$HOME/idiotypeforge" && git pull --quiet
tmux kill-session -t forge 2>/dev/null || true
tmux new-session -d -s forge "
  cd $HOME/idiotypeforge
  bash scripts/setup_a100.sh                 2>&1 | tee -a ~/run.log
  source .env.a100
  bash scripts/run_gcp_a100.sh fixtures      2>&1 | tee -a ~/run.log
  bash scripts/run_gcp_a100.sh finetune      2>&1 | tee -a ~/run.log
  echo '━━━ DONE ━━━' | tee -a ~/run.log
  exec bash
"
echo "✓ tmux session 'forge' detached and running."
EOF
```

### C. Monitor

```bash
# 5. Tail the log from a fresh SSH (any time)
gcloud compute ssh idiotypeforge-a100-spot --zone=$ZONE -- tail -f ~/run.log

# 6. Or attach to tmux directly
gcloud compute ssh idiotypeforge-a100-spot --zone=$ZONE -- tmux attach -t forge

# 7. Check GPU utilisation in real time
gcloud compute ssh idiotypeforge-a100-spot --zone=$ZONE -- watch -n 5 nvidia-smi
```

### D. Pull results back + tear down

```bash
# 8. After "━━━ DONE ━━━", scp the artefacts back
gcloud compute scp --zone=$ZONE --recurse \
  idiotypeforge-a100-spot:~/idiotypeforge/data/demo_cases ./data/
gcloud compute scp --zone=$ZONE --recurse \
  idiotypeforge-a100-spot:~/idiotypeforge/runs ./

# Or just have the VM push to git itself:
gcloud compute ssh idiotypeforge-a100-spot --zone=$ZONE -- bash <<'EOF'
cd ~/idiotypeforge
git config user.email aniruddh.goteti@orbion.life
git config user.name "Aniruddh Goteti"
git add -A && git commit -m "GPU outputs from $(date +%Y-%m-%d) A100 run"
git push
EOF

# 9. ALWAYS DELETE THE VM WHEN DONE
gcloud compute instances delete idiotypeforge-a100-spot --zone=$ZONE --quiet
```

### E. Cost expectations

| GPU | Spot $/hr (incl. VM) | Wallclock | Total |
|---|---|---|---|
| L4 (g2-standard-8) | ~$0.38 | ~10 hr | ~$4 — *but heavily preempted on this date* |
| A100 40 GB (a2-highgpu-1g) | ~$1.15 | ~6 hr | **~$7–10** ⭐ |
| A100 80 GB | n/a | — | quota = 0 in this project |

Recommended: **A100 spot, ~$10 budget**. If it gets preempted (DELETE
mode means disk is gone), re-run sections A → D from scratch. Total wasted
on a single preemption: ~$0.50 + 5 min.

---

## 7. Hugging Face Hub setup (Day 14)

The Unsloth fine-tune script can push the LoRA adapter to HF Hub if
`HF_TOKEN` and `HUB_ID` are set. Steps:

```bash
# On your laptop (one-time): create the empty repo
huggingface-cli login    # paste your token
huggingface-cli repo create idiotypeforge-gemma4-e4b-ab-lora --type model

# On the VM, before running finetune:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
export HUB_ID=YOUR_HF_USERNAME/idiotypeforge-gemma4-e4b-ab-lora
bash scripts/run_gcp_a100.sh finetune
```

If `HF_TOKEN` is unset, the adapter saves locally to
`runs/gemma4-e4b-ab-lora/lora/` and the script prints a warning.

---

## 8. What can be done WITHOUT GPU (in the meantime)

These all advance the submission and need zero cloud spend:

1. **Render the Excalidraw architecture diagram** — open
   `docs/architecture.excalidraw` at <https://excalidraw.com>, export
   to `docs/architecture.png` and `docs/architecture.svg`, commit. The
   README references the PNG and currently shows a broken-image icon.

2. **Polish the writeup** at `docs/writeup.md` — currently a skeleton.
   Target: ≤1500 words. Use the personal opener from the README.

3. **Pre-record the demo footage** — Gradio UI runs locally
   (`uv run python -m app.ui.gradio_app`). Click through the three demo
   cases, screen-record at 60 fps. The agent log streams in real time.
   Even with mocks, this is the footage the video uses; the GPU run
   only changes the *backing data*, not the user-visible flow.

4. **Run the setup tests on a fresh clone** — verifies the README's
   quickstart actually works for a new user:
   ```bash
   cd /tmp && git clone https://github.com/aniruddhgoteti/idiotypeforge.git
   cd idiotypeforge && bash scripts/setup_local.sh
   ```

5. **Write the cover image** in Figma per
   `docs/video_shotlist.md` "Cover image / thumbnail" section.

---

## 9. Open decisions / known limitations

- **The orchestrator's Gemma mode is implemented but the dossier
  composer's Gemma mode is a stub** that falls back to the template.
  Wiring up real Ollama-driven dossier composition is a Day 7-extension
  job, ~2 hours of work. For now, template mode produces a complete
  provenance-clean dossier so we don't strictly need it.

- **The fine-tune evaluates perplexity but doesn't yet run the
  CDR3-masked-AA top-1 accuracy benchmark** specified in the plan §4.
  Adding that to `scripts/finetune_gemma4_unsloth.py` is a small task
  if we want to claim the plan's exact threshold ("≥ 50 % vs ~25 %
  base").

- **The Gradio UI's saliency tab is stubbed** — `app/ui/saliency.py`
  has the AbLang2 attention rollup helper, but there's no UI panel
  using it yet. Day 12 work.

- **Off-target tool depends on `scripts/download_oas.py` having been
  run.** When it hasn't, the orchestrator gracefully falls back to a
  zero-identity placeholder (correctly flagged in the dossier). The
  download is ~50 MB, takes 1 minute on a good connection. Run it on
  the laptop OR on the VM during setup_a100.sh.

- **`docs/A100_RUNBOOK.md` was planned but I never wrote it.** Section 6
  above is the de-facto runbook. If we want a standalone doc, it's a
  copy-paste of §6 into `docs/A100_RUNBOOK.md`.

---

## 10. Commits on `main` (most recent first)

```
0e1f4cf  Day 13-14: GPU utilisation optimisations for L4/A100 spot VMs
7c5ae12  Day 13-14 GPU code: real RFdiffusion + ColabFold + Unsloth
d0cce44  Day-bonus: dose estimator + YouTube shotlist
aa7d9d8  Days 8 + 10 + 11: UI dashboard, end-to-end harness, decision cards
0995595  Days 6-7: real demo BCR sequences + full orchestrator wiring
af26e11  Replace ASCII architecture diagram with Excalidraw source
8406c57  README: personal lymphoma story + beginner module walkthrough
1e9237b  Days 4-5: MHCflurry, structure renderer, dossier composer
0ce934d  Days 2-3: real CPU tools, verification gates, data download
a5beb4a  Reframe project around access equity, not novelty
69481b3  Day 1: scaffold IdiotypeForge with CPU-first methodology
```

Tests: 110 pass, 7 intentionally skipped.

---

## 11. Hackathon submission checklist (rolling)

When picking up later, the deadline is **18 May 2026 at 11:59 PM UTC**.
Required for submission:

- [ ] Kaggle Writeup (≤1500 words) — `docs/writeup.md` skeleton exists
- [ ] **Public Code Repo** — ✅ <https://github.com/aniruddhgoteti/idiotypeforge>
- [ ] **3-min YouTube video** — shotlist at `docs/video_shotlist.md`
- [ ] **Live Demo URL** — needs HF Space deploy (Day 15)
- [ ] **Cover image** — Figma per shotlist specs
- [ ] Track selection in Writeup — Main + Health & Sciences + Digital Equity (+ Unsloth if fine-tune ships)

Judging weights (Kaggle):
- 40 % Impact & Vision
- 30 % Video Pitch & Storytelling  ← biggest leverage point
- 30 % Technical Depth & Execution

The video matters more than the code does. Spend at least one full day
on it.

---

## 12. How to bootstrap the new chat

Paste this whole file as the first message in the new chat, then add
your specific question. The new assistant should:

1. **Read this file fully** before doing anything.
2. **Verify the repo state** — `cd ~/workspace/idiotypeforge && git log
   --oneline | head -5` should show `0e1f4cf` at the top.
3. **Run the tests** — `python3 -m pytest tests/ -q` should produce
   "110 passed, 7 skipped".
4. **Acknowledge the personal frame** in §2 — do not flatten it into
   statistics.

Then proceed with whatever you need: GPU run, video, writeup, deploy,
new feature.
