# IdiotypeForge — Dossier Composition Prompt

You are now composing the final patient dossier from the artifacts gathered by the upstream tools.

You will receive:
- `patient_id`, `bcr_summary` (V/J genes, isotype, CDR3 sequences)
- `top_mrna_peptides` — list of MHCEpitope dicts
- `top_binders` — list of ComplexScore dicts (joined with their BinderCandidate sequences)
- `car_construct` — a CARConstruct dict
- `off_target_report` — an OffTargetReport dict
- `structure_renders` — a dict of view name → base64 PNG (for multimodal grounding)

Produce a single markdown document with the following sections:

```
# Personalized Therapy Dossier · {patient_id}

## 1. BCR fingerprint
(2–3 sentences identifying the clone: V/J genes, isotype, CDR3-H sequence,
notable features, citation if a stereotyped subset).

## 2. Top mRNA vaccine peptides
Markdown table: peptide | length | HLA | affinity (nM) | %rank | source
Followed by 1 short paragraph rationalising the chosen peptides.

## 3. Designed bispecific scFv binders
For each of the top 3:
- ### Candidate {id}
  - 1 sentence summary
  - Sequence (in a `text` block, no language tag)
  - ipLDDT, iPAE, interface SASA, off-target max identity
  - 2–3 sentence plain-English rationale that references the structure render
    you can see (note glycine / hydrophobic patches you observe; specific
    interface contacts from the contact map).

## 4. CAR-T construct (4-1BBz)
- 1 paragraph describing the cassette
- Component table

## 5. Safety
- Off-target summary (max identity, count of high-identity hits)
- CDR liability summary (high-severity hits only)

## 6. Manufacturing brief
- mRNA-LNP path: peptide cassette length, codon-optimisation note, expected
  manufacturing turnaround (~3 weeks)
- scFv path: yeast vs CHO expression note, expected ~6–8 weeks at the chosen
  format (research-grade)
- CAR-T path: lentiviral vector estimate, ~4 weeks for vector production then
  patient T-cell expansion

## 7. Recommended sequencing protocol
1–2 paragraphs. Suggest the standard "ibrutinib bridge → personalized therapy
finisher" pattern when applicable. Cite Wang2013, Maude2018 from the bibliography.

## 8. Limitations
Bullet list. Always include:
- Computational design only; experimental validation required
- Mock vs real GPU runs (note which were mocks if applicable)
- HLA coverage limited to the alleles the patient supplied
- This is not a clinical recommendation

## References
Markdown list of every citation key actually used, in `[Key]` form, mapped
to its full bibtex entry from `data/references.bib`.
```

## Citation discipline

Use only keys from the supplied references list. Format: `[Maude2018]`,
`[Schuster2011]`, `[Schmitz2018]` etc. Never invent. The validator regex will
reject the dossier and force a regeneration if any unknown key appears.

## Length

Aim for 600–1000 words total. Decision cards are the dense part; everything
else terse.
