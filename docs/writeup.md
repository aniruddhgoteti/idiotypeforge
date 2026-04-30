# IdiotypeForge — Kaggle Writeup (DRAFT)

> **DRAFT.** Final pass on Day 16. Target: ≤ 1 500 words.

## Title (working)

**IdiotypeForge: AI-designed personalized anti-idiotype therapy for B-cell lymphoma**

## Subtitle

A Gemma 4 agent collapses personalized cancer therapy design from six months to under two hours.

---

## 1. The problem (50 words)

Standard lymphoma care is uneven: 88 % five-year survival in well-resourced Hodgkin patients vs ~30 % in PTCL, <20 % in ATLL, ~30 % in primary CNS, ~6 months median OS post-CAR-T relapse, <30 % Burkitt survival in sub-Saharan Africa. Personalised therapy exists — for the lucky. This project is for everyone else.

## 2. Why BiovaxID failed — and why that failure is an equity story (150 words)

[TODO: BiovaxID, FavId, MyVax 2009–2011 — biology was right, manufacturing logistics killed it. 3–6-month per-patient turnaround, $100K+ cost, trial design enrolled patients in remission. Cite Schuster2011, Levy2014. **Crucial framing**: the $100K+ price tag and 6-month timeline didn't just kill the trials — they pre-emptively excluded every healthcare system that couldn't absorb that. The wall protected the lucky and condemned the unlucky. Breaking it has to be cheap and local, or it isn't broken at all.]

## 3. The five-tech wall now broken (200 words)

[TODO: AlphaFold/IgFold (Jumper2021, Ruffolo2023), RFdiffusion (Watson2023), ProteinMPNN (Dauparas2022), mRNA-LNP manufacturing post-COVID, OAS scale (Olsen2022), native-tool-calling LLMs (Gemma 4). Each individually insufficient; the combination changes the regime.]

## 4. IdiotypeForge architecture (300 words)

[TODO: pipeline diagram from README. Nine deterministic tools + Gemma 4 native function calling. Multimodal vision over rendered Fv structures. Long-context dossier synthesis. Domain-adapted Unsloth QLoRA on OAS antibodies. Local Ollama deployment for hospital PHI. Mock-mode for local methodology validation, real GPU phase only for fixture pre-compute and fine-tune.]

## 5. Three case studies (300 words)

[TODO: per case (FL, CLL, DLBCL): patient summary → top mRNA peptides → top binders → off-target metrics → decision card highlight → 1-paragraph rationale referencing the structure render.]

## 6. Benchmarks (200 words)

[TODO: comparison table from plan §6 — ibrutinib chronic suppression cost / CD19 CAR-T escape rate / BiovaxID 6-month baseline / IdiotypeForge target. Insert per-case wallclock numbers from notebooks/02_benchmarks.ipynb.]

## 7. Gemma 4 + the access argument (200 words)

[TODO: function calling × 9 tools, multimodal × 3 structure views per case, long context for dossier, Unsloth QLoRA fine-tune on OAS + 30% perplexity reduction. **Lean hard on on-prem via Ollama**: closed APIs can't be deployed behind hospital firewalls in jurisdictions without HIPAA-equivalent or where bandwidth is the bottleneck. Show the LiteRT/llama.cpp story for sub-A100 GPUs. The whole point of building on open weights is that a haematology unit in Kampala or Manila can run this without a cloud bill.]

## 8. Limitations and what comes next (100 words)

[TODO: research artifact disclaimer. Wet-lab validation required. HLA coverage limited to supplied alleles. Mock vs. real GPU runs noted in dossier. Manufacturing handoff to mRNA-LNP / scFv expression partners. Regulatory framework: FDA platform designation precedent. **Equity roadmap**: which collaborators (PTCL trials, African Burkitt cohorts, primary CNS lymphoma networks) we'd hand this to first.]

---

## Citations

Pulled from `data/references.bib` — never hand-typed.

[TODO: render bibliography in submission style.]
