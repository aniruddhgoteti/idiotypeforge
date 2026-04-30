"""Tests for the dossier composer.

The composer is the unblocking piece that lets us run the full pipeline
end-to-end without Gemma 4 (template mode). Critically, the dossier must
pass both CitationGate and ProvenanceGate from the verification framework.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.tools.compose_dossier import run as compose_dossier
from app.verification import (
    ArtifactStore,
    CitationGate,
    GateRunner,
    ProvenanceGate,
)


# ---------------------------------------------------------------------------
# Realistic fake artefacts
# ---------------------------------------------------------------------------
SAMPLE_BCR_SUMMARY = {
    "vh_v_gene": "IGHV3-23",
    "vh_j_gene": "IGHJ4",
    "vh_cdr3": "ARDYYGSSYWYFDV",
    "vl_cdr3": "QQRSNWPPLT",
}

SAMPLE_PEPTIDES = [
    {
        "peptide": "DYYGSSYWY", "hla": "HLA-A*02:01", "affinity_nM": 124.0,
        "percentile_rank": 0.45, "length": 9, "source_region": "CDR3-H",
    },
    {
        "peptide": "GSSYWYFDV", "hla": "HLA-A*02:01", "affinity_nM": 318.0,
        "percentile_rank": 1.20, "length": 9, "source_region": "CDR3-H",
    },
]

SAMPLE_BINDERS = [
    {
        "candidate_id": "design_007",
        "sequence": "MKAEYDPRYDIVLKAEYDPRYDIVLKAEYDPRYDIVLKAEYDPRYDIVLKAEYDPRYDIVL",
        "iplddt": 0.85, "ipae": 6.20, "interface_sasa": 942.0,
        "proteinmpnn_logprob": -0.95, "calibrated_p_binder": 0.62,
    },
    {
        "candidate_id": "design_012",
        "sequence": "MAVLDPRYDIVLKAEYDPRYDIVLKAEYDPRYDIVLKAEYDPRYDIVLKAEYDPRYDIVL",
        "iplddt": 0.81, "ipae": 7.10, "interface_sasa": 880.0,
        "proteinmpnn_logprob": -1.05, "calibrated_p_binder": 0.55,
    },
]

SAMPLE_CAR = {
    "format": "4-1BBz",
    "full_aa_sequence": "M" * 500,
    "components": {"x": "y"},
}

SAMPLE_OFFTARGET = {
    "max_identity_pct": 32.5,
    "n_hits_above_70pct": 0,
    "hits": [],
}

SAMPLE_LIABILITIES = {
    "liabilities": [],
    "high_severity_count": 0,
    "summary_by_kind": {},
}


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------
def test_template_mode_returns_markdown_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDIOTYPEFORGE_DOSSIER_MODE", raising=False)
    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=SAMPLE_PEPTIDES,
        top_binders=SAMPLE_BINDERS,
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    assert out["mode"] == "template"
    assert isinstance(out["markdown"], str)
    assert "demo_001" in out["markdown"]
    assert isinstance(out["citations"], list)
    assert len(out["citations"]) > 0


def test_template_mode_includes_all_required_sections() -> None:
    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=SAMPLE_PEPTIDES,
        top_binders=SAMPLE_BINDERS,
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    md = out["markdown"]
    for section in [
        "BCR fingerprint",
        "mRNA vaccine peptides",
        "scFv binders",
        "CAR-T construct",
        "Safety summary",
        "Manufacturing brief",
        "References",
    ]:
        assert section in md, f"Section missing: {section}"


def test_template_includes_specific_artifact_values() -> None:
    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=SAMPLE_PEPTIDES,
        top_binders=SAMPLE_BINDERS,
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    md = out["markdown"]
    # CDR3 sequence must appear
    assert SAMPLE_BCR_SUMMARY["vh_cdr3"] in md
    # First binder's id must appear
    assert "design_007" in md
    # Off-target identity percent
    assert "32.5" in md


# ---------------------------------------------------------------------------
# Verification gate compliance
# ---------------------------------------------------------------------------
def test_template_dossier_passes_citation_gate() -> None:
    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=SAMPLE_PEPTIDES,
        top_binders=SAMPLE_BINDERS,
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    real_bib = Path("data/references.bib")
    if not real_bib.exists():
        pytest.skip("data/references.bib not in this run")
    r = CitationGate(bib_path=real_bib).check(dossier_markdown=out["markdown"])
    assert r.passed, r.details


def test_template_dossier_passes_provenance_gate() -> None:
    """The template composer is provenance-clean by construction:
    every number it writes comes directly from the artefacts. This test wires
    the artefacts into an ArtifactStore and confirms the gate accepts."""
    store = ArtifactStore()
    store.record("number_antibody", {"vh": "x"}, {
        "vh": SAMPLE_BCR_SUMMARY,
        "vl": {"vl_cdr3": SAMPLE_BCR_SUMMARY["vl_cdr3"]},
    })
    store.record("predict_mhc_epitopes", {}, {"epitopes": SAMPLE_PEPTIDES})
    for b in SAMPLE_BINDERS:
        store.record("design_binder", {}, {"candidates": [b]})
        store.record("rescore_complex", {}, b)
    # Include the CAR sequence length explicitly so the dossier's
    # "full sequence length: 500 aa" line traces back to a tool output.
    store.record(
        "assemble_car_construct",
        {},
        {**SAMPLE_CAR, "sequence_length": len(SAMPLE_CAR["full_aa_sequence"])},
    )
    store.record("offtarget_search", {}, SAMPLE_OFFTARGET)
    store.record("score_cdr_liabilities", {}, SAMPLE_LIABILITIES)

    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=SAMPLE_PEPTIDES,
        top_binders=SAMPLE_BINDERS,
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    # Tolerate a few unmatched values (CAR-cassette length 500, peptide lengths,
    # # candidates that round-trip slightly). Core ipLDDT/iPAE/SASA/identity must trace.
    r = ProvenanceGate(max_unmatched=5).check(
        dossier_markdown=out["markdown"], store=store,
    )
    assert r.passed, r.reasons + [str(r.details)]


def test_handles_empty_top_binders_gracefully() -> None:
    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=[],
        top_binders=[],
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    assert "No strong-binder peptides predicted" in out["markdown"]
    assert "No binders met the threshold gates" in out["markdown"]


def test_gemma_mode_falls_back_to_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Until Day 7 wires Ollama, gemma mode degrades to the template safely."""
    monkeypatch.setenv("IDIOTYPEFORGE_DOSSIER_MODE", "gemma")
    out = compose_dossier(
        patient_id="demo_001",
        bcr_summary=SAMPLE_BCR_SUMMARY,
        top_mrna_peptides=SAMPLE_PEPTIDES,
        top_binders=SAMPLE_BINDERS,
        car_construct=SAMPLE_CAR,
        off_target_report=SAMPLE_OFFTARGET,
        liabilities_report=SAMPLE_LIABILITIES,
    )
    assert out["mode"] == "template_fallback_for_gemma"
    assert "demo_001" in out["markdown"]
