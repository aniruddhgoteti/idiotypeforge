"""Unit tests for the deterministic tool wrappers.

Local CPU runs by default. Tests marked `gpu` are skipped unless CUDA is
available. Tests marked `slow` run real IgFold/MHCflurry/etc. on CPU and
take minutes — skipped from `pytest -k 'not slow'`.
"""
from __future__ import annotations

import os

import pytest

from app.agent.router import dispatch, list_tools
from app.tools import car_assembler
from app.tools._mocks import use_mocks


# ---------------------------------------------------------------------------
# Router & registry
# ---------------------------------------------------------------------------
def test_router_lists_all_tools() -> None:
    expected = {
        "number_antibody",
        "predict_fv_structure",
        "score_cdr_liabilities",
        "predict_mhc_epitopes",
        "design_binder",
        "rescore_complex",
        "offtarget_search",
        "assemble_car_construct",
        "render_structure",
        "compose_dossier",
    }
    assert set(list_tools()) == expected


def test_router_unknown_tool_returns_error() -> None:
    out = dispatch("totally_made_up_tool", {})
    assert "error" in out
    assert "available" in out


# ---------------------------------------------------------------------------
# Mocks (the GPU tools)
# ---------------------------------------------------------------------------
def test_mocks_default_on() -> None:
    assert use_mocks() is True, "Mocks should be ON by default for local dev."


def test_design_binder_mock_returns_n_designs() -> None:
    out = dispatch(
        "design_binder",
        {
            "target_pdb": "REMARK fake pdb\nEND",
            "hotspot_residues": [95, 96, 97, 98, 99, 100],
            "n_designs": 5,
        },
    )
    assert "candidates" in out
    assert len(out["candidates"]) == 5
    assert out["mock"] is True
    # Sorted by log-prob descending
    lps = [c["proteinmpnn_logprob"] for c in out["candidates"]]
    assert lps == sorted(lps, reverse=True)


def test_design_binder_is_deterministic() -> None:
    args = {
        "target_pdb": "REMARK seed pdb\nEND",
        "hotspot_residues": [101, 102],
        "n_designs": 3,
    }
    a = dispatch("design_binder", args)
    b = dispatch("design_binder", args)
    assert a["candidates"] == b["candidates"], "Mock must be deterministic per input."


def test_rescore_complex_mock_shape() -> None:
    out = dispatch(
        "rescore_complex",
        {
            "binder_sequence": "MKAEYDPRYDIVL" * 5,
            "target_pdb": "REMARK\nEND",
            "candidate_id": "design_000",
        },
    )
    for k in ("iplddt", "ipae", "interface_sasa", "contact_count"):
        assert k in out
    assert 0.0 <= out["iplddt"] <= 1.0
    assert out["ipae"] >= 2.0
    assert out["mock"] is True


# ---------------------------------------------------------------------------
# CAR assembler (deterministic, no stubs)
# ---------------------------------------------------------------------------
def test_car_assembler_4_1bbz() -> None:
    fake_vh = "EVQLVQSGGGLVKPGGSLRLSCAASGFTFSSYGMHWVRQAPG"
    fake_vl = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGK"
    out = car_assembler.run(scfv_vh=fake_vh, scfv_vl=fake_vl, format="4-1BBz")
    assert out["format"] == "4-1BBz"
    full = out["full_aa_sequence"]
    assert fake_vh in full
    assert fake_vl in full
    assert out["components"]["linker"] == "GGGGSGGGGSGGGGS"
    # CD3z must be the C-terminal block
    assert full.endswith(out["components"]["CD3z"])


def test_car_assembler_rejects_bad_format() -> None:
    with pytest.raises(ValueError):
        car_assembler.run(scfv_vh="X", scfv_vl="X", format="invalid")


# ---------------------------------------------------------------------------
# Stub tools — confirm they error gracefully (so the router surface is testable)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tool, args",
    [
        ("number_antibody", {"vh_sequence": "EVQ", "vl_sequence": "DIQ"}),
        ("predict_fv_structure", {"vh_sequence": "EVQ", "vl_sequence": "DIQ"}),
        ("score_cdr_liabilities", {"vh_numbering": {}, "vl_numbering": {}}),
        ("predict_mhc_epitopes", {
            "cdr3_h_aa": "ARDYYGSSYWYFDV",
            "cdr3_l_aa": "QQRSNWPPLT",
            "hla_alleles": ["HLA-A*02:01"],
        }),
        ("offtarget_search", {"query_sequence": "ARDYYGSSYWYFDV"}),
        ("render_structure", {"pdb_text": "REMARK\nEND"}),
        ("compose_dossier", {
            "patient_id": "demo",
            "bcr_summary": {},
            "top_mrna_peptides": [],
            "top_binders": [],
            "car_construct": {},
            "off_target_report": {},
        }),
    ],
)
def test_stubs_return_structured_error(tool: str, args: dict) -> None:
    out = dispatch(tool, args)
    assert isinstance(out, dict)
    assert "error" in out
    assert out["error"] == "stub_not_implemented"
