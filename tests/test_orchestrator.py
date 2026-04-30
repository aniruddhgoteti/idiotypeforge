"""Tests for the orchestrator.

Template-mode is fully exercised here: the agent runs end-to-end without
any LLM, calling the 9 tools in fixed order. The pipeline must produce a
non-empty dossier and pass the verification gates.

Gemma-mode tests are limited to error-handling / fallback paths since
Ollama isn't installed in CI.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.agent.orchestrator import (
    AgentEvent,
    PatientInput,
    _read_fasta_pair,
    _redact,
    _summarise,
    dispatch_traced,
    run_agent,
    verify_dossier,
)
from app.verification import ArtifactStore


# ---------------------------------------------------------------------------
# Fixture: a patient that doesn't depend on ANARCI being installed.
# We patch number_antibody to return a synthetic numbering object.
# ---------------------------------------------------------------------------
HAVE_ANARCI = importlib.util.find_spec("anarci") is not None


SYNTHETIC_NUMBERING = {
    "vh": {
        "chain_type": "H",
        "scheme": "kabat",
        "v_gene": "IGHV3-23",
        "j_gene": "IGHJ4",
        "isotype": None,
        "cdr1": {"start": 26, "end": 32, "sequence": "GFTFSSY"},
        "cdr2": {"start": 50, "end": 58, "sequence": "ISSSGGSTY"},
        "cdr3": {"start": 95, "end": 108, "sequence": "ARDYYGSSYWYFDV"},
        "framework_sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSCAVRSAY",
    },
    "vl": {
        "chain_type": "K",
        "scheme": "kabat",
        "v_gene": "IGKV1-39",
        "j_gene": "IGKJ4",
        "isotype": None,
        "cdr1": {"start": 24, "end": 32, "sequence": "RASQSVSSY"},
        "cdr2": {"start": 50, "end": 56, "sequence": "DASNRAT"},
        "cdr3": {"start": 89, "end": 98, "sequence": "QQRSNWPPLT"},
        "framework_sequence": "DIQMTQSPSSLSASVGDRVTITCRSGSGCSY",
    },
}

SYNTHETIC_STRUCTURE = {
    "pdb_text": (
        "ATOM      1  CA  ALA H   1      10.000  10.000  10.000  1.00 90.00\n"
        "ATOM      2  CA  GLY H   2      11.000  10.000  10.000  1.00 88.00\n"
        "ATOM      3  CA  ALA L   1      10.000  12.000  10.000  1.00 86.00\n"
        "ATOM      4  CA  ASP L   2      11.000  12.000  10.000  1.00 84.00\n"
        "END\n"
    ),
    "plddt": [0.9, 0.88, 0.86, 0.84],
    "mean_plddt": 0.87,
    "cdr3_mean_plddt": 0.78,
    "render_png_b64": None,
    "wallclock_seconds": 0.1,
}

SYNTHETIC_EPITOPES = {
    "epitopes": [
        {"peptide": "DYYGSSYWY", "hla": "HLA-A*02:01", "affinity_nM": 124.0,
         "percentile_rank": 0.45, "length": 9, "source_region": "CDR3-H"},
    ],
    "n_evaluated": 12,
    "n_strong_binders": 1,
}


def _patient(patient_id: str = "test_001") -> PatientInput:
    return PatientInput(
        patient_id=patient_id,
        vh_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDYYGSSYWYFDVWGQGTLVTVSS",
        vl_sequence="DIQMTQSPSSLSASVGDRVTITCRASQSVSSYLAWYQQKPGKAPKLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPPLTFGGGTKVEIK",
        hla_alleles=["HLA-A*02:01"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_redact_truncates_long_strings() -> None:
    out = _redact({"vh_sequence": "A" * 200, "patient_id": "demo", "n": 5})
    assert out["vh_sequence"].startswith("<")
    assert "200-char" in out["vh_sequence"]
    assert out["patient_id"] == "demo"
    assert out["n"] == 5


def test_summarise_caps_string_length() -> None:
    s = _summarise({"x": "y" * 1000}, max_chars=100)
    assert len(s) <= 101    # 100 + ellipsis
    assert s.endswith("…")


def test_dispatch_traced_records_into_store() -> None:
    store = ArtifactStore()
    out = dispatch_traced("assemble_car_construct",
                          {"scfv_vh": "EVQ", "scfv_vl": "DIQ", "format": "4-1BBz"},
                          store)
    assert "full_aa_sequence" in out
    assert len(store) == 1
    assert store.artifacts[0].tool_name == "assemble_car_construct"


def test_read_fasta_pair_parses_two_records(tmp_path: Path) -> None:
    p = tmp_path / "fv.fasta"
    p.write_text(">case|VH\nEVQLVQ\nGGSL\n>case|VL\nDIQMTQ\nSPSS\n")
    vh, vl = _read_fasta_pair(p)
    assert vh == "EVQLVQGGSL"
    assert vl == "DIQMTQSPSS"


# ---------------------------------------------------------------------------
# Template-mode end-to-end (uses synthetic patches for ANARCI/IgFold/MHCflurry
# so the test runs on the base install with no heavy deps)
# ---------------------------------------------------------------------------
def _patched_dispatch(name: str, args):       # type: ignore[no-untyped-def]
    """Drop-in for `dispatch` that returns canned outputs for not-yet-installed tools.

    Falls through to the real router for everything that's already implemented.
    """
    if name == "number_antibody":
        return SYNTHETIC_NUMBERING
    if name == "predict_fv_structure":
        return SYNTHETIC_STRUCTURE
    if name == "predict_mhc_epitopes":
        return SYNTHETIC_EPITOPES
    # everything else (cdr_liabilities, design_binder, rescore_complex,
    # offtarget_search, assemble_car_construct, render_structure,
    # compose_dossier) is implemented or has a deterministic mock.
    from app.agent.router import dispatch as real_dispatch
    return real_dispatch(name, args)


def test_template_mode_runs_end_to_end_and_produces_dossier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    monkeypatch.setattr("app.agent.orchestrator.dispatch", _patched_dispatch)

    events = list(run_agent(_patient(), mode="template"))
    assert events[0].kind == "thought"
    assert events[-1].kind == "final"

    final = events[-1].payload
    assert isinstance(final, dict)
    assert final["mode"] == "template"
    assert final["dossier_markdown"]              # non-empty markdown
    assert "Personalized Therapy Dossier" in final["dossier_markdown"]
    assert final["n_tool_calls"] >= 8


def test_template_mode_emits_a_tool_call_for_each_pipeline_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    monkeypatch.setattr("app.agent.orchestrator.dispatch", _patched_dispatch)

    events = list(run_agent(_patient(), mode="template"))
    tool_call_names = [
        e.payload["name"] for e in events
        if e.kind == "tool_call" and isinstance(e.payload, dict)
    ]
    # We expect at least these (some are called multiple times — e.g. rescore per candidate)
    expected = {
        "number_antibody",
        "predict_fv_structure",
        "score_cdr_liabilities",
        "predict_mhc_epitopes",
        "design_binder",
        "rescore_complex_batch",   # batched call instead of per-candidate
        "offtarget_search",
        "assemble_car_construct",
        "estimate_doses",
        "compose_dossier",
    }
    assert expected.issubset(set(tool_call_names))


def test_template_mode_dossier_passes_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    monkeypatch.setattr("app.agent.orchestrator.dispatch", _patched_dispatch)

    events = list(run_agent(_patient(), mode="template"))
    final = events[-1].payload
    assert final["verification_passed"], final["audit_markdown"]


def test_template_mode_handles_missing_offtarget_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """If offtarget_search is still a stub (data not downloaded), pipeline still completes."""
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")

    # Patch dispatch to make offtarget_search return a stub error so we
    # can check the orchestrator's graceful fallback path.
    def patched(name, args):                  # type: ignore[no-untyped-def]
        if name == "offtarget_search":
            return {"error": "stub_not_implemented", "detail": "x"}
        return _patched_dispatch(name, args)

    monkeypatch.setattr("app.agent.orchestrator.dispatch", patched)
    events = list(run_agent(_patient(), mode="template"))
    final = events[-1].payload
    assert isinstance(final, dict)
    assert final["dossier_markdown"]            # still produced


# ---------------------------------------------------------------------------
# Gemma-mode fallback path
# ---------------------------------------------------------------------------
def test_gemma_mode_falls_back_to_template_when_ollama_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `ollama` package isn't importable, gemma mode degrades gracefully."""
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    monkeypatch.setattr("app.agent.orchestrator.dispatch", _patched_dispatch)

    # Force the import of `ollama` to fail by injecting a None into sys.modules
    import sys
    real_ollama = sys.modules.pop("ollama", None)
    sys.modules["ollama"] = None              # type: ignore[assignment]
    try:
        events = list(run_agent(_patient(), mode="gemma"))
    finally:
        if real_ollama is not None:
            sys.modules["ollama"] = real_ollama
        else:
            sys.modules.pop("ollama", None)

    final = events[-1].payload
    assert isinstance(final, dict)
    assert final["mode"] == "template"        # fell back


# ---------------------------------------------------------------------------
# verify_dossier alone
# ---------------------------------------------------------------------------
def test_verify_dossier_returns_audit_markdown() -> None:
    store = ArtifactStore()
    store.record("rescore_complex", {}, {"iplddt": 0.85})
    out = verify_dossier(
        dossier_markdown="ipLDDT 0.85 — see [Maude2018].",
        store=store,
    )
    assert "audit_markdown" in out
    assert "MockModeGate" in out["audit_markdown"]
