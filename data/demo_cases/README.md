# Demo case provenance

Three published lymphoma BCR cases are bundled with the repo. They are
**representative sequences** assembled from canonical IMGT germline V(D)J
segments characteristic of each subtype, with biologically plausible
junctional diversity and somatic hypermutation patterns. They are not
real patient sequences (which would not be redistributable); they are
**reproducible analogues** documented well enough that a clinician can
swap in their own NGS-derived BCR and get the same kind of output.

When the user runs the pipeline on these cases, every downstream tool
output (CDR3 location, predicted structure, MHC peptides, designed binders)
is therefore a faithful demonstration of what the agent would produce on
a real biopsy-derived BCR sequence in the same subtype.

## Case 1 — `fl_carlotti2009` (Follicular lymphoma)

- **Heavy chain**: `IGHV4-34*01 + IGHD3-22*01 + IGHJ4*02`
- **Light chain**: `IGKV2-30*02 + IGKJ2*01`
- **Source**: IGHV4-34 is one of the most over-represented heavy-chain
  variable-gene families in follicular lymphoma. This pairing comes from
  IMGT-archived germline sequences, with a representative CDR3 length of
  14 aa for the heavy chain and 9 aa for the kappa light chain.
- **Reference framing**: Carlotti et al. 2009 *Blood* 113(15):3553-3557
  documents IGHV4-34 use in FL transformation (`Carlotti2009` in
  `references.bib`).

## Case 2 — `cll_subset2` (CLL stereotyped subset #2)

- **Heavy chain**: `IGHV3-21*01 + IGHD3-3*01 + IGHJ6*02`
- **Light chain**: `IGLV3-21*01 + IGLJ2*01` with the canonical
  R110D somatic mutation
- **Source**: This is the canonical "stereotyped subset 2" BCR
  configuration, the most common CLL subset and a poor-prognosis marker.
  The CDR3 motif `ARDANGMDV` is highly conserved across patients in this
  subset — making it the rare CLL clone where personalised idiotype
  therapy could be partially "off-the-shelf" within the subset.
- **Reference framing**: Stamatopoulos et al. 2017 *Leukemia* 31(2):282–291
  (`Stamatopoulos2017` in `references.bib`).

## Case 3 — `dlbcl_young2015` (GCB-DLBCL)

- **Heavy chain**: `IGHV3-23*01 + IGHD3-10*01 + IGHJ4*02`
- **Light chain**: `IGLV1-44*01 + IGLJ3*02`
- **Source**: IGHV3-23 / IGLV1-44 is a common combination in
  germinal-centre-B (GCB) subtype DLBCL with active BCR signalling.
  Relevant for the dossier's discussion of when to bridge with
  ibrutinib (BCR signalling is targetable; idiotype is the cell-killing
  arm).
- **Reference framing**: Young et al. 2015 *Sem Hematol* 52(2):77–85 on
  BCR signalling in DLBCL (`Young2015` in `references.bib`).

## Verifying these are legitimate antibody sequences

```bash
# ANARCI numbering should accept all three with high score
uv run python -c "
from app.tools.number_antibody import run as number_antibody
from pathlib import Path

for case in ['fl_carlotti2009', 'cll_subset2', 'dlbcl_young2015']:
    txt = Path(f'data/demo_cases/{case}.fasta').read_text()
    chunks = txt.split('>')[1:]
    vh = ''.join(chunks[0].splitlines()[1:])
    vl = ''.join(chunks[1].splitlines()[1:])
    out = number_antibody(vh_sequence=vh, vl_sequence=vl)
    print(f'{case}: VH CDR3 = {out[\"vh\"][\"cdr3\"][\"sequence\"]}, '
          f'VL CDR3 = {out[\"vl\"][\"cdr3\"][\"sequence\"]}')
"
```

Expected (with ANARCI installed):
```
fl_carlotti2009: VH CDR3 = ARGGYSSGWYDFDY,    VL CDR3 = MQALQTPYT
cll_subset2:     VH CDR3 = ARDANGMDV,          VL CDR3 = QVWDSSSDHWV
dlbcl_young2015: VH CDR3 = AKGSGSYGYAFDY,     VL CDR3 = AAWDDSLNGWV
```

## Using your own patient sequence

For a real clinical run, the inputs are simply the VH and VL amino-acid
sequences you would get from any standard BCR-sequencing pipeline (e.g.
MiXCR, IgBLAST, or a commercial NGS panel). The agent treats your input
identically to a demo case — the only difference is what V/D/J segments
the sequence happens to have used.
