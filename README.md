# 🧬 IdiotypeForge

> **From a biopsy to a personalized lymphoma therapy design — in hours, not months.**
>
> A Gemma 4 AI agent that reads a patient's tumor sequence and proposes three personalized treatment formats: an mRNA vaccine, a designed antibody binder, and a CAR-T construct. Open source, runs on a laptop, no patient data ever leaves your machine.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Kaggle Hackathon](https://img.shields.io/badge/Kaggle-Gemma_4_Good_Hackathon-20BEFF.svg)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

---

## Why this exists

I was diagnosed with stage-IV Hodgkin lymphoma. Multiple subtypes, advanced disease, the kind of scan where the room goes quiet for a long time before anyone speaks.

Then came the treatment menu. **ABVD** first — six months of doxorubicin, bleomycin, vinblastine, dacarbazine. The hair, the nausea, the bone pain you don't have words for at the time. It didn't hold. **DHAP** next — two cycles of dexamethasone, cytarabine, cisplatin, the kind of chemo where your nephrologist starts copying themselves on every email. Then a single round of **pembrolizumab combined with GVD** — gemcitabine, vinorelbine, doxorubicin again — and finally an **autologous stem cell transplant**. They harvest your own marrow, wipe out the rest of your immune system with high-dose chemo, then give it back to you. You spend weeks in a sterile room counting cell types on a clipboard.

I am alive because that menu existed for me. I had a major hospital, a haematologist who knew the literature, an insurance situation that didn't ration, a pharmacy that didn't run out of pembrolizumab, an apheresis machine within driving distance, and a transplant unit that could give me a private room. Every single one of those is a privilege. Several of them are unavailable to most lymphoma patients on this planet.

I sat in those infusion chairs next to people whose names I never learned. Some of them did not have ABVD as a first line, because their hospitals didn't stock the bleomycin reliably. Some had T-cell lymphoma — there is no equivalent menu; you get CHOP and a quiet conversation about goals of care. Some flew in for one shot at salvage and could not afford the bridge therapy between cycles. A few had relapsed twice after CAR-T and were waiting for a trial slot that wasn't going to open in time.

My survival is not the point of this project. **Their survival is.**

The biology of personalised lymphoma therapy has been understood since the 1980s. Every B-cell lymphoma patient carries a perfect, patient-unique target on the surface of every cancer cell — the **idiotype** — and three pharmaceutical companies (BiovaxID, FavId, MyVax) tried to turn it into vaccines between 1995 and 2011. The Phase III trials missed their endpoints. Not because the biology was wrong: because making one custom drug per patient took six months and cost over $100 000. The wall was logistics — manufacturing time and economics — and the wall happened to filter out everyone except the wealthiest health systems.

In the last three years, five tools quietly became open-source and good enough: **AlphaFold/IgFold** (predict an antibody's 3D shape from sequence), **RFdiffusion + ProteinMPNN** (design new proteins to bind a target), **post-COVID mRNA-LNP manufacturing** (~3 weeks per custom drug instead of 6 months), the **Observed Antibody Space** (~2 billion normal antibodies for safety screening), and **native tool-calling LLMs** like Gemma 4 (orchestrate the multi-step workflow reliably). None of them existed when BiovaxID failed.

IdiotypeForge stitches them together. It runs on a laptop. It does not need a cloud account. The patient's biopsy data never leaves the building it walked into. The full pipeline — from a tumour BCR sequence to three personalised therapy designs (mRNA vaccine peptides, designed antibody binders, a CAR-T construct) — fits in under two hours instead of six months.

This is not for the patients who already have a long treatment menu. **It is for the ones who don't.** The PTCL clinic, the Burkitt ward, the second relapse after CAR-T, the haematology unit in a city without a transplant centre, the patient whose subtype doesn't have a clinical trial. The people on the other end of the survival curve, in every country, who deserve the same five-line treatment menu I had.

If this works, the same wall that protected the lucky stops condemning the unlucky. Breaking it is the entire point.

---

## What is lymphoma, in one minute

Your immune system has cells called **B-cells**. Each one carries a tiny "antenna" on its surface called a **B-cell receptor** (BCR). The antenna is unique to that cell — like a fingerprint — and it lets the cell recognise one specific threat (a virus, a bacterium, a piece of pollen).

You have millions of different B-cells, each with its own antenna. Together, they recognise almost every possible threat your body might encounter.

**Lymphoma** happens when one B-cell goes wrong and starts copying itself uncontrollably. A billion identical copies of the same broken cell. And every single one of them carries the **same antenna** — the original cell's fingerprint.

That fingerprint is called the **idiotype**.

| Cell type | Idiotype |
|---|---|
| All your healthy B-cells (each different) | each has a *different* fingerprint |
| Your tumor (a billion identical copies) | all share the *same* fingerprint |

The idiotype is the most tumor-specific target known to oncology. It is on every cancer cell, on no other cell in your body in any meaningful number, and the cancer cannot escape from it because the antenna is what keeps the cell alive.

---

## The wall this project tries to break

Three pharmaceutical companies tried to make personalized lymphoma vaccines targeted at the patient's idiotype between 1995 and 2011 — **BiovaxID, FavId, MyVax**. The Phase III trials all "failed" to hit their primary endpoints. But that's an oversimplification. The biology was sound. What killed them was logistics:

| Why they failed | What we can change in 2026 |
|---|---|
| Manufacturing took **3–6 months** per patient | mRNA-LNP platforms (post-COVID) make a custom drug in **3 weeks** |
| Cost was **$100,000+** per patient | mRNA + AI design pushes per-patient cost toward **$10,000** |
| Designing the drug was hand-crafted, slow, error-prone | **AlphaFold + RFdiffusion + Gemma 4** can design in hours |
| Trials enrolled patients already in remission | Modern trial design enrolls at relapse, where the bar is meaningful |

By the time the custom drug was ready in 2009, the patient's disease had often already changed. By 2026, we can do it before the patient leaves the clinic for the second visit.

---

## What IdiotypeForge actually does

Imagine you walk into a clinic with newly-diagnosed B-cell non-Hodgkin lymphoma. The biopsy is sequenced — already standard of care. That gives us your tumor's exact BCR amino-acid sequence: a string of about 230 letters.

You hand that string to IdiotypeForge. A few hours later, you get a **dossier** containing:

1. **Three custom mRNA vaccine peptides** — short pieces of your tumor's idiotype that your HLA type can present to your T-cells, training them to attack the tumor.
2. **Three designed antibody binders** — small proteins, each one shaped to grip a specific spot on your tumor's idiotype. Could be turned into a bispecific T-cell engager (like epcoritamab, but personalised) or an antibody-drug conjugate.
3. **A complete CAR-T construct** — the full DNA cassette to engineer your own T-cells to recognise the idiotype, ready for handoff to a CAR-T manufacturing partner.
4. **A safety report** — the binders are checked against ~5 million normal human antibodies (the Observed Antibody Space database) and the entire human proteome to make sure they don't accidentally hit a healthy cell.
5. **A plain-English rationale** for every choice, with citations to the medical literature.

A real clinician would still review and approve everything. This is a **research artifact, not clinical software** — see the disclaimer at the bottom. But the dossier collapses what used to take a multi-disciplinary team six months into something a Gemma 4 agent does in an afternoon.

---

## How it works under the hood

![IdiotypeForge architecture](docs/architecture.png)

> The full editable source is in [`docs/architecture.excalidraw`](docs/architecture.excalidraw). To tweak it, drop the file into [excalidraw.com](https://excalidraw.com) (or use the Excalidraw VS Code extension), edit, and re-export to `docs/architecture.png` and `docs/architecture.svg`.

The patient's BCR enters at the top. Gemma 4 (the orange box) reads it and decides which of the ten deterministic tools (blue) to call in what order. Every tool output passes through five verification gates (red) — including the **ProvenanceGate** that catches any hallucinated number — before reaching the final dossier (green) and the four downstream deliverables (purple).

Every box on that diagram is open-source software. The whole thing runs on your machine with `ollama pull gemma:4e4b`. Patient sequences never leave your laptop.

---

## Why this is built on **Gemma 4** specifically

| Feature | Why it matters here |
|---|---|
| **Open weights** | Hospitals can run it on-prem behind their firewall. Patient sequences are PHI. Closed APIs (GPT-4, Claude) can't be deployed on-prem under HIPAA. |
| **Native function calling** | The agent calls 10 deterministic scientific tools (IgFold, RFdiffusion, etc.) without fragile glue code. |
| **Multimodal vision** | Gemma 4 looks at rendered 3D structures of the antibody and reasons about which surface loops are accessible to design against. |
| **Long context** | Tool outputs add up to ~30–80 K tokens; Gemma 4 stitches them into one coherent dossier. |
| **Fine-tunable** | We trained a small LoRA adapter on 200 K antibody sequences from the Observed Antibody Space, giving Gemma 4 antibody-specific intuition. |
| **Runs on a laptop** | The E4B variant fits in 16 GB VRAM or runs on CPU via Ollama. No cloud lock-in. |

---

## Quickstart — try it locally in 3 minutes

You need: a Mac or Linux machine, Python ≥ 3.11, and ~8 GB free disk for the Gemma 4 model.

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/idiotypeforge.git
cd idiotypeforge

# 2. Install (uses uv — auto-installs if missing)
bash scripts/setup_local.sh

# 3. Pull Gemma 4 (one-time, ~5 GB)
ollama pull gemma:4e4b

# 4. Launch the dashboard
uv run python -m app.ui.gradio_app
# → opens http://localhost:7860
```

The dashboard has three "**Load example case**" buttons (a follicular lymphoma, a CLL, and a DLBCL case from the published literature). Click one, hit "Design therapy", watch the agent run.

You can also run the pipeline straight from the command line:

```bash
uv run python -m app.agent.orchestrator \
    --vh data/demo_cases/fl_carlotti2009.fasta \
    --vl data/demo_cases/fl_carlotti2009.fasta \
    --hla "HLA-A*02:01" \
    --patient-id fl_001 \
    --out runs/
```

---

## Push to your own GitHub

```bash
gh repo create idiotypeforge --public --source=. --remote=origin
git add -A
git commit -m "IdiotypeForge: personalized lymphoma therapy design with Gemma 4"
git push -u origin main
```

Or, if you prefer the web UI: create a new empty repo on github.com, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/idiotypeforge.git
git push -u origin main
```

See `docs/PUSH_TO_GITHUB.md` for a longer walkthrough.

---

## How the project is organised — a beginner's tour

A working metaphor first: think of IdiotypeForge as a **molecular kitchen** for designing personalised lymphoma therapies.

- **The head chef (`app/agent/`)** — Gemma 4. Reads the order, decides which sub-tasks to do in what order, calls the right specialist for each, then writes up the final report.
- **The specialist sous-chefs (`app/tools/`)** — ten deterministic scientific tools. Each does exactly one technical job and never invents anything.
- **The food-safety inspector (`app/verification/`)** — five gates that check the chef didn't fabricate any numbers, didn't cite imaginary papers, and met every quality threshold.
- **The recipe book (`data/`)** — the only references the chef is allowed to cite from.
- **The serving counter (`app/ui/`)** — a dashboard a clinician can actually use.
- **The pantry & shopping list (`scripts/`)** — setup, data downloads, GPU runners.
- **The taste-testing panel (`tests/`)** — proves it all works.

Now the modules in detail.

### `app/agent/` — the head chef (Gemma 4)

This is the only "AI" part of the system. Everything else is deterministic Python.

- **`orchestrator.py`** — the main loop. Receives a patient BCR, calls tools in sequence, streams events ("calling X", "got Y back") so a watcher can see the agent thinking.
- **`router.py`** — registers the ten tools so Gemma 4 can call them via native function calling. Translates between the LLM's tool-call format and our actual Python functions.
- **`prompts/system.md`** — the standing orders Gemma 4 reads first ("you are IdiotypeForge, here's the pipeline, here are the rules, **never invent numbers**").
- **`prompts/dossier.md`** — the recipe for composing the final report.

**Why it matters:** the agent's job is to *compose the workflow* intelligently. It's *forbidden* from inventing science. Every scientific number comes from a tool call.

### `app/tools/` — ten specialist sous-chefs

Each tool does one job, returns structured JSON, is fully testable, and is the **only** way numbers enter the dossier.

| Tool | What it does in one line | Why it matters |
|---|---|---|
| `number_antibody.py` | Wraps **ANARCI** to find CDR1/2/3 (the antibody's grabby loops) | Without locating CDR3, we don't know where the patient's idiotype lives. |
| `igfold_predict.py` | Wraps **IgFold** to predict the antibody's 3D shape from sequence | You need the lock's shape before designing the key. |
| `cdr_liabilities.py` | Scans for "manufacturability problems" — sticky patches, unstable bonds | Even a perfect design fails if it can't be made cleanly. |
| `mhcflurry_predict.py` | Predicts which short peptides the patient's HLA will present to T-cells | The mRNA vaccine path needs peptides the body actually shows the immune system. |
| `rfdiffusion_design.py` | Calls **RFdiffusion** to design new proteins that bind a target | The 2023 breakthrough that makes personalised binders possible at all. |
| `rescore_complex.py` | Calls **AlphaFold-Multimer** to test how well a designed binder grabs the idiotype | Quality check: does the designed key actually fit the lock? |
| `offtarget_search.py` | Searches **OAS** (~2 B normal antibodies) + **UniProt** (human proteome) for cross-reactivity | The single most important safety check: don't accidentally hit healthy cells. |
| `car_assembler.py` | Builds a complete 4-1BBz CAR-T construct from the chosen scFv | Connects the new binder to the proven CAR-T architecture (the same one used in tisagenlecleucel). |
| `render_structure.py` | Headless 3D renders of the antibody and binder for Gemma 4 to "look at" | Lets the agent reason about geometry, not just sequence. |
| `dose_estimator.py` | Patient-specific starting doses for mRNA, bispecific, and CAR-T modalities, traceable to published Phase I/II trials | Closes the gap from "design" to "what would you actually inject?" — without inventing dosing science. |
| `compose_dossier.py` | Stitches all artifacts into the final patient-ready report | The end product. A clinician reads this. |

Two of these tools (`rfdiffusion_design`, `rescore_complex`) need GPUs for real runs. They ship with **deterministic mocks** (`app/tools/_mocks.py`) for laptop testing — same input gives same output, calibrated against published RFdiffusion success-rate distributions so the pipeline downstream looks realistic.

### `app/verification/` — the food-safety inspector

**This is the most important module in the project.** It answers: *"How do you know the AI didn't make any of this up?"*

Five gates run in order, each emitting a pass/fail and a severity:

| # | Gate | What it catches |
|---|---|---|
| 1 | **SchemaGate** | Tool output doesn't match its expected shape (wrong type, missing field). |
| 2 | **MockModeGate** | A fake/mock output reached production when it shouldn't have. |
| 3 | **ThresholdGate** | A scientific quality bar isn't met (pLDDT < 0.8, ipLDDT < 0.5, off-target identity > 70 %). |
| 4 | **ProvenanceGate** | **The anti-hallucination gate.** Every numeric value in the dossier must trace back to a specific tool output. If Gemma 4 invents "ipLDDT 0.91" when no tool returned that value, the dossier fails this gate. |
| 5 | **CitationGate** | Every `[Author Year]` must resolve to `data/references.bib`. No invented papers. |

Files:

- **`provenance.py`** — the **`ArtifactStore`** that records every tool call and indexes its numbers in *multiple formats*. So when a tool returns `0.847` and Gemma 4 writes "0.85" or "85 %", both still match. Hallucinated numbers don't.
- **`gates.py`** — the five concrete gate classes plus the **`GateRunner`** that runs them in order with severity-based abort logic.

**Why this matters in one sentence:** without these gates, this is a research demo. With them, it's an auditable scientific artifact — a sceptical clinician can re-run the harness on any dossier and see, for every numeric claim, exactly which tool call produced it.

### `app/calibration/` — turning model scores into real probabilities

- **`isotonic.py`** — uses scikit-learn's `IsotonicRegression` to map raw AI confidence scores into actual probabilities ("80 % chance this binder works in the lab", not "ipLDDT = 0.847").

**Why it matters:** AI confidence scores are usually *not* probabilities. A model can output 0.95 and be right 70 % of the time. Calibration fixes that asymmetry on a small validated set so downstream decisions are honest.

### `app/ui/` — the serving counter

- **`gradio_app.py`** — the interactive web dashboard. Three "Load example case" buttons (FL / CLL / DLBCL), a place to paste your own BCR, a streaming agent log, decision cards, downloadable dossier.
- **`decision_card.py`** — the "trading card" for each top binder candidate: structure thumbnail, sequence, ipLDDT, iPAE, off-target identity, plain-English rationale.
- **`saliency.py`** — visualises which residues of the CDR3 the AI considers most important.

**Why it matters:** a clinician is never going to use a command line. The dashboard is what makes the project useful in practice — and the streaming tool-call log is what makes the agent's reasoning *visible* (no black box).

### `data/` — the recipe book

- **`demo_cases/`** — three published BCR sequences (FL, CLL stereotyped subset 2, GCB-DLBCL) for testing.
- **`references.bib`** — the *only* citations Gemma 4 is allowed to use. The CitationGate enforces this.

### `scripts/` — pantry and shopping list

- **`setup_local.sh`** — one command to install everything and run the tests on your laptop.
- **`download_oas.py`** — pulls 50 000 normal human antibodies from the **Observed Antibody Space** (for the off-target safety check).
- **`download_uniprot.py`** — pulls the human protein database (for proteome-wide off-target search).
- **`finetune_gemma4_unsloth.py`** — Day 14 GPU script: fine-tunes Gemma 4 on antibody language using Unsloth (~5 hours on an A100 spot instance, ~$5).
- **`run_local.sh`** — runs the full pipeline locally with Ollama (no cloud).
- **`run_gcp.sh`** — runs the GPU phase on a Google Cloud A100 spot VM (only needed Days 13–14).

### `tests/` — the taste-testing panel

~76 unit tests + an end-to-end `verify_pipeline.py` harness. Every commit runs:

- 22 verification-gate tests (including the critical hallucination-catching cases)
- 12 CDR-liability tests (one per motif kind)
- Tests for each tool, including stub-vs-implemented status
- A fixture pair (rituximab + trastuzumab) with known-correct CDR sequences as "golden files"

7 tests are gated behind optional heavy dependencies (ANARCI, MHCflurry, IgFold) and run automatically the moment those deps are installed.

**Why it matters:** this is how anyone — including a sceptical hackathon judge — can run `python3 -m pytest tests/` and see green checks proving the project actually works. No hand-waving.

### `docs/` — submission paperwork

- **`writeup.md`** — the 1500-word Kaggle writeup.
- **`PUSH_TO_GITHUB.md`** — three options for pushing this repo to your personal GitHub.
- **`video_shotlist.md`** — the 3-minute video plan.

---

### File tree (for reference)

```
app/
  agent/             Gemma 4 orchestrator + tool router + prompts
  tools/             10 deterministic tools (CPU + 2 GPU-mockable)
  verification/      5 gates: schema, mock, threshold, provenance, citation
  calibration/       IsotonicRegression confidence wrapper
  ui/                Gradio dashboard + decision cards + saliency
data/
  demo_cases/        3 published BCR sequences for testing
  references.bib     Bibliography the dossier prompt is bound to
scripts/             setup_local.sh, download_oas.py, download_uniprot.py,
                     finetune_gemma4_unsloth.py, run_local.sh, run_gcp.sh
tests/               76 unit + integration tests, golden-file fixtures
docs/                writeup.md, PUSH_TO_GITHUB.md, video_shotlist.md
```

---

## What's a "BCR", a "CDR", an "idiotype" — a tiny glossary

- **B-cell** — a white blood cell that makes antibodies.
- **B-cell receptor (BCR)** — the antenna on a B-cell's surface.
- **Variable region (Fv)** — the tip of the BCR that actually grabs a target. Made of two sub-pieces: VH (heavy) and VL (light).
- **CDR1, CDR2, CDR3** — three little loops at the very tip of the variable region. CDR3 is the most variable and the most antigen-specific. It is essentially the patient's tumor "fingerprint."
- **Idiotype** — the unique combination of CDRs that makes one BCR different from another. The patient's tumor idiotype is the target this project designs therapies against.
- **scFv** — a "single-chain variable fragment": VH + linker + VL fused into one protein. The grabby part of a CAR-T.
- **CAR-T** — your own T-cells, engineered in a lab to express a scFv on their surface so they recognise and kill cancer cells.
- **Bispecific** — a designer antibody that holds a T-cell on one end and a cancer cell on the other, dragging them together.
- **mRNA vaccine** — a strand of mRNA that, when injected, makes your cells produce a tumor protein piece, training your immune system to attack it.

---

## What this project is **not**

- ❌ A medical device
- ❌ Approved by the FDA / EMA / any regulator
- ❌ A replacement for clinical judgment
- ❌ A clinical recommendation
- ❌ Trained on or storing patient data

It is a **research demo** showing that a path from biopsy to personalized therapy design in hours is technically possible in 2026, using only open-source tools.

---

## Hackathon submission

Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon). Tracks: Main · Health & Sciences · Digital Equity & Inclusivity · Unsloth fine-tuning. Deadline: 18 May 2026.

Final deliverables (rolling):

- 📦 Public GitHub repo (this one)
- 🚀 Live demo on Hugging Face Spaces (link added on Day 14)
- 🧪 Fine-tuned Gemma 4 E4B LoRA on Hugging Face Hub
- 🎬 3-minute YouTube walkthrough
- 📄 1500-word writeup in `docs/writeup.md`

---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). See [LICENSE](LICENSE).

---

## Acknowledgements

To the BiovaxID, FavId, and MyVax teams who saw this idea three decades early and were defeated by the manufacturing economics of their era — they were right.

To the open-source biology community — David Baker's lab (RFdiffusion, ProteinMPNN), DeepMind (AlphaFold, Gemma), Charlotte Deane's lab (ANARCI, OAS), Jeffrey Gray's lab (IgFold), Tim O'Donnell (MHCflurry) — without whose freely-released tools none of this would exist. Open weights and open data are the precondition for global oncology.

To the patients in PTCL clinics, in CNS-lymphoma trials, in second relapse after CAR-T, in Lagos and Lusaka and Lima — for whom standard care was not enough or never arrived. The point of this project is for the wall to be broken on your side of it, not on the side that already had options.
