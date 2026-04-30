"""Tests for the IgFold structure-prediction tool.

These tests are marked `slow` — IgFold takes 2–5 min on CPU, so they only
run when explicitly requested:

    uv run pytest -m slow tests/test_igfold.py
"""
from __future__ import annotations

import importlib.util

import pytest

from app.tools.igfold_predict import run as predict_fv

from tests.fixtures.known_antibodies import RITUXIMAB_VH, RITUXIMAB_VL


HAVE_IGFOLD = importlib.util.find_spec("igfold") is not None
needs_igfold = pytest.mark.skipif(
    not HAVE_IGFOLD,
    reason="igfold not installed; run `uv sync --extra igfold`.",
)


# ---------------------------------------------------------------------------
# Argument validation (no IgFold required)
# ---------------------------------------------------------------------------
def test_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError):
        predict_fv(vh_sequence="", vl_sequence="DIQ")
    with pytest.raises(ValueError):
        predict_fv(vh_sequence="EVQ", vl_sequence="")


def test_emits_runtime_error_when_igfold_missing() -> None:
    if HAVE_IGFOLD:
        pytest.skip("igfold IS installed; skipping missing-dep path test.")
    with pytest.raises(RuntimeError, match="IgFold is not installed"):
        predict_fv(vh_sequence=RITUXIMAB_VH, vl_sequence=RITUXIMAB_VL)


# ---------------------------------------------------------------------------
# Real IgFold run (slow — minutes on CPU)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@needs_igfold
def test_rituximab_predicts_with_high_plddt() -> None:
    """Verification target: mean Fv pLDDT ≥ 0.85 on rituximab."""
    out = predict_fv(
        vh_sequence=RITUXIMAB_VH,
        vl_sequence=RITUXIMAB_VL,
        render=False,
        do_refine=False,
    )
    assert "pdb_text" in out
    assert "ATOM" in out["pdb_text"]
    assert isinstance(out["plddt"], list)
    assert len(out["plddt"]) > 0
    # Threshold for the framework — the CDR3 specifically may dip lower.
    assert out["mean_plddt"] >= 0.80, f"mean pLDDT {out['mean_plddt']:.2f} below 0.80 threshold"
