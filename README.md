# 🧬 IdiotypeForge

> **From a biopsy to a personalized lymphoma therapy design — in hours, not months.**
>
> A Gemma 4 AI agent that reads a patient's tumor sequence and proposes three personalized treatment formats: an mRNA vaccine, a designed antibody binder, and a CAR-T construct. Open source, runs on a laptop, no patient data ever leaves your machine.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Kaggle Hackathon](https://img.shields.io/badge/Kaggle-Gemma_4_Good_Hackathon-20BEFF.svg)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

---

## Who this is for

In the United States, a Hodgkin lymphoma patient diagnosed in 2024 has a roughly **88 % five-year survival rate**. They will be offered ABVD, escalated BEACOPP, brentuximab-AVD, salvage DHAP or ICE, pembrolizumab combinations, autologous stem cell transplant, sometimes CAR-T. The treatments are gruelling, but the menu is long, and the outcomes are reasonable.

This project is not for those patients. It is for the people one or two doors down the ward. The people on the other end of the survival curve.

| Population | 5-year survival | What's missing |
|---|---|---|
| Peripheral T-cell lymphoma (PTCL-NOS) | **~ 30 %** | Almost no targeted therapy; treated with a 1970s B-cell regimen |
| Adult T-cell leukaemia/lymphoma (ATLL) | **< 20 %** | Endemic in regions with the lowest oncology budgets |
| Primary CNS lymphoma | **~ 30 %** | Most CAR-T and bispecifics don't cross the blood-brain barrier |
| Relapsed/refractory DLBCL after CAR-T | median OS **~ 6 months** | CD19 escape; no good next-line target |
| Richter transformation (CLL → DLBCL) | median OS **6–12 months** | No reliable targeted therapy |
| Double-hit / triple-hit lymphoma | 5-yr OS **~ 30 %** | MYC + BCL2 are still considered undruggable |
| Burkitt lymphoma in low-resource settings | **< 30 %** in sub-Saharan Africa, **vs > 85 %** in high-income countries | Same biology, different ZIP code |
| Any lymphoma patient in a country without CAR-T infrastructure | varies | Personalised therapy is geographically locked |

Standard care, where it exists, is among the great triumphs of modern medicine. But it is not evenly distributed across subtypes, across treatment lines, or across the planet.

IdiotypeForge is built on the bet that **personalised lymphoma therapy should not require a $475 000 CAR-T infrastructure or a six-month design cycle**. The biology of patient-specific targeting was understood in 1985. The reason it never reached the people who needed it most was logistics — manufacturing time and cost. Five technologies that didn't exist three years ago (AlphaFold/IgFold, RFdiffusion, post-COVID mRNA-LNP manufacturing, the Observed Antibody Space, and tool-calling LLMs like Gemma 4) collectively make those logistics breakable in 2026.

This project does not replace ABVD, R-CHOP, CAR-T, or pembrolizumab. It is for the gap those treatments cannot reach: **the patient with a poor-prognosis subtype, in a low-resource setting, after standard therapy has run out**. It runs on a single laptop, behind a hospital firewall, with no cloud dependency. The patient's biopsy data never leaves the building.

If we are right, the same wall that protected the lucky also condemned the unlucky. Breaking it is the point.

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

## How it works under the hood (one diagram)

```
       Patient BCR sequence (VH + VL)
                   │
                   ▼
       ┌──────── Gemma 4 ────────┐
       │  multimodal LLM agent   │
       │  with native tool use   │
       └──────────┬──────────────┘
                  │
   ┌──────────────┼─────────────────────┐
   ▼              ▼                     ▼
ANARCI       IgFold (CPU/GPU)      AbLang2
numbering    structure prediction  embeddings
   │              │                     │
   └──────────────┼─────────────────────┘
                  ▼
      ┌──────── targeting ────────┐
      │   MHCflurry: epitopes     │  → mRNA peptides
      │   RFdiffusion + ProteinMPNN: binders │  → bispecifics / CAR scFv
      └──────────────┬────────────┘
                     ▼
       AlphaFold-Multimer rescore  →  ipLDDT, iPAE, calibrated P(binder)
                     ▼
       MMseqs2 + BLAST off-target  →  safety against healthy human Ig + proteome
                     ▼
       CAR-T construct assembler   →  4-1BBz cassette ready for synthesis
                     ▼
       Gemma 4 dossier composition →  decision cards + citations
```

Every box on that diagram is open-source software. The whole thing runs on your machine with `ollama pull gemma:4e4b`. Patient sequences never leave your laptop.

---

## Why this is built on **Gemma 4** specifically

| Feature | Why it matters here |
|---|---|
| **Open weights** | Hospitals can run it on-prem behind their firewall. Patient sequences are PHI. Closed APIs (GPT-4, Claude) can't be deployed on-prem under HIPAA. |
| **Native function calling** | The agent calls 9 deterministic scientific tools (IgFold, RFdiffusion, etc.) without fragile glue code. |
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

## Repository layout

```
app/
  agent/            Gemma 4 orchestrator and tool router
  tools/            Nine deterministic tools the agent calls
  ui/               Gradio dashboard, decision cards, saliency
  calibration/      Confidence calibration (IsotonicRegression)
data/
  demo_cases/       Three published patient BCRs for quick testing
  references.bib    Bibliography the dossier prompt is bound to
scripts/            Local setup, OAS download, GCP runner, Unsloth fine-tune
tests/              Unit + end-to-end verification harness
docs/               Hackathon writeup, video shotlist, GitHub guide
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
