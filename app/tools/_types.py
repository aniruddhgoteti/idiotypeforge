"""Shared dataclasses and type aliases used across tool modules.

Keep these pydantic models small and JSON-serialisable so the agent can pass
them directly to / from Gemma 4 function calls without custom encoders.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Antibody numbering
# ---------------------------------------------------------------------------
class CDRSpan(BaseModel):
    """Inclusive 1-based residue indices for a CDR loop."""

    start: int
    end: int
    sequence: str


class ChainNumbering(BaseModel):
    """ANARCI numbering output for a single Ig chain."""

    chain_type: Literal["H", "K", "L"]
    scheme: Literal["kabat", "imgt", "chothia"] = "kabat"
    v_gene: Optional[str] = None
    j_gene: Optional[str] = None
    isotype: Optional[str] = None
    cdr1: CDRSpan
    cdr2: CDRSpan
    cdr3: CDRSpan
    framework_sequence: str


# ---------------------------------------------------------------------------
# Structure prediction
# ---------------------------------------------------------------------------
class FvStructure(BaseModel):
    """IgFold / ESMFold output for a paired Fv region."""

    pdb_text: str
    plddt: list[float] = Field(..., description="Per-residue pLDDT in [0, 1]")
    mean_plddt: float
    cdr3_mean_plddt: float
    render_png_b64: Optional[str] = None


# ---------------------------------------------------------------------------
# CDR liability scan
# ---------------------------------------------------------------------------
class Liability(BaseModel):
    """A developability liability motif found in a CDR or framework."""

    kind: Literal[
        "n_glycosylation",      # NXS/T
        "deamidation",          # NG, NS
        "isomerization",        # DG, DS, DT, DD, DH
        "oxidation",            # M, W
        "free_cysteine",
        "fragmentation",        # DP
    ]
    chain: Literal["H", "L"]
    region: Literal["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    position: int               # 1-based within the chain
    motif: str
    severity: Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# MHC epitope prediction
# ---------------------------------------------------------------------------
class MHCEpitope(BaseModel):
    """A predicted HLA-restricted peptide from CDR3."""

    peptide: str
    hla: str                    # e.g. "HLA-A*02:01"
    affinity_nM: float          # predicted IC50
    percentile_rank: float
    length: int
    source_region: Literal["CDR3-H", "CDR3-L"]


# ---------------------------------------------------------------------------
# Binder design (RFdiffusion + ProteinMPNN)
# ---------------------------------------------------------------------------
class BinderCandidate(BaseModel):
    """A designed binder against the patient's idiotype."""

    candidate_id: str
    sequence: str
    length: int
    proteinmpnn_logprob: float
    scaffold_pdb: str
    designed_against_hotspots: list[int]


# ---------------------------------------------------------------------------
# Complex rescore (AlphaFold-Multimer / Boltz-2)
# ---------------------------------------------------------------------------
class ComplexScore(BaseModel):
    """Predicted-complex quality metrics."""

    candidate_id: str
    iplddt: float               # interface pLDDT in [0, 1]
    ipae: float                 # interface PAE (Å)
    interface_sasa: float       # buried SASA (Å²)
    contact_count: int
    calibrated_p_binder: Optional[float] = None  # IsotonicRegression output


# ---------------------------------------------------------------------------
# Off-target safety
# ---------------------------------------------------------------------------
class OffTargetHit(BaseModel):
    """A near-match to the candidate epitope/binder in normal repertoire."""

    database: Literal["OAS_healthy_paired", "UniProt_human"]
    hit_id: str
    identity_pct: float
    coverage_pct: float
    e_value: Optional[float] = None


class OffTargetReport(BaseModel):
    max_identity_pct: float
    n_hits_above_70pct: int
    hits: list[OffTargetHit]


# ---------------------------------------------------------------------------
# CAR-T construct
# ---------------------------------------------------------------------------
class CARConstruct(BaseModel):
    """Assembled CAR-T cassette ready for plasmid synthesis."""

    format: Literal["CD28z", "4-1BBz"]
    full_aa_sequence: str
    components: dict[str, str]   # leader, scFv_VH, linker, scFv_VL, hinge, TM, costim, CD3z


# ---------------------------------------------------------------------------
# Final dossier
# ---------------------------------------------------------------------------
class TherapyDossier(BaseModel):
    """The full design output Gemma 4 composes for a patient."""

    patient_id: str
    bcr_summary: dict
    top_mrna_peptides: list[MHCEpitope]
    top_binders: list[ComplexScore]   # joined w/ BinderCandidate by candidate_id
    car_construct: CARConstruct
    off_target_report: OffTargetReport
    rationale_markdown: str
    citations: list[str]              # bibtex keys from references.bib
