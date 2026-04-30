# IdiotypeForge — Kaggle Writeup (DRAFT)

> **DRAFT.** Final pass on Day 16. Target: ≤ 1 500 words.

## Title (working)

**IdiotypeForge: AI-designed personalized anti-idiotype therapy for B-cell lymphoma**

## Subtitle

A Gemma 4 agent collapses personalized cancer therapy design from six months to under two hours.

---

## 1. The problem (50 words)

Every B-cell non-Hodgkin lymphoma patient carries the most tumor-specific antigen in oncology — their tumor's own B-cell receptor (BCR), known as the *idiotype*. It is on every cancer cell, on almost no healthy cell, and the cancer cannot mutate away from it without dying.

## 2. Why BiovaxID failed (150 words)

[TODO: BiovaxID, FavId, MyVax 2009–2011 — biology was right, manufacturing logistics killed it. 3–6-month per-patient turnaround, $100K+ cost, trial design enrolled patients in remission. Cite Schuster2011, Levy2014.]

## 3. The five-tech wall now broken (200 words)

[TODO: AlphaFold/IgFold (Jumper2021, Ruffolo2023), RFdiffusion (Watson2023), ProteinMPNN (Dauparas2022), mRNA-LNP manufacturing post-COVID, OAS scale (Olsen2022), native-tool-calling LLMs (Gemma 4). Each individually insufficient; the combination changes the regime.]

## 4. IdiotypeForge architecture (300 words)

[TODO: pipeline diagram from README. Nine deterministic tools + Gemma 4 native function calling. Multimodal vision over rendered Fv structures. Long-context dossier synthesis. Domain-adapted Unsloth QLoRA on OAS antibodies. Local Ollama deployment for hospital PHI. Mock-mode for local methodology validation, real GPU phase only for fixture pre-compute and fine-tune.]

## 5. Three case studies (300 words)

[TODO: per case (FL, CLL, DLBCL): patient summary → top mRNA peptides → top binders → off-target metrics → decision card highlight → 1-paragraph rationale referencing the structure render.]

## 6. Benchmarks (200 words)

[TODO: comparison table from plan §6 — ibrutinib chronic suppression cost / CD19 CAR-T escape rate / BiovaxID 6-month baseline / IdiotypeForge target. Insert per-case wallclock numbers from notebooks/02_benchmarks.ipynb.]

## 7. Gemma 4 specifics (200 words)

[TODO: function calling × 9 tools, multimodal × 3 structure views per case, long context for dossier, Unsloth QLoRA fine-tune on OAS + 30% perplexity reduction, on-prem via Ollama (PHI argument), why closed models can't do this.]

## 8. Limitations and what comes next (100 words)

[TODO: research artifact disclaimer. Wet-lab validation required. HLA coverage limited to supplied alleles. Mock vs. real GPU runs noted in dossier. Manufacturing handoff to mRNA-LNP / scFv expression partners. Regulatory framework: FDA platform designation precedent.]

---

## Citations

Pulled from `data/references.bib` — never hand-typed.

[TODO: render bibliography in submission style.]
