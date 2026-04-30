"""Tests for the MHCflurry epitope predictor.

The real predictor needs ~150 MB of model weights. Tests are split:
  - argument validation + window enumeration: always run
  - real prediction call: skipped unless MHCflurry + models are installed
"""
from __future__ import annotations

import importlib.util

import pytest

from app.tools.mhcflurry_predict import run as predict_epitopes, slide_windows


HAVE_MHCFLURRY = importlib.util.find_spec("mhcflurry") is not None
needs_mhcflurry = pytest.mark.skipif(
    not HAVE_MHCFLURRY,
    reason="mhcflurry not installed; run `uv pip install mhcflurry` and "
           "`mhcflurry-downloads fetch models_class1_presentation`.",
)


# ---------------------------------------------------------------------------
# Window enumeration (no external deps)
# ---------------------------------------------------------------------------
def test_slide_windows_basic() -> None:
    out = slide_windows("ARDYYGSSY", [8, 9])
    # 8-mers: ARDYYGSS, RDYYGSSY = 2
    # 9-mers: ARDYYGSSY = 1
    assert len(out) == 3
    assert "ARDYYGSSY" in out


def test_slide_windows_dedupes() -> None:
    out = slide_windows("AAAAA", [4])
    assert out == ["AAAA"]    # only one unique 4-mer


def test_slide_windows_skips_noncanonical_chars() -> None:
    out = slide_windows("ARXYY*GS", [3])
    # any kmer containing X or * is dropped
    assert all("X" not in k and "*" not in k for k in out)


def test_slide_windows_skips_short_input() -> None:
    """If sequence shorter than k, no windows."""
    assert slide_windows("AB", [9]) == []


# ---------------------------------------------------------------------------
# Argument validation (no MHCflurry call required)
# ---------------------------------------------------------------------------
def test_rejects_empty_cdr3() -> None:
    with pytest.raises(ValueError):
        predict_epitopes(cdr3_h_aa="", cdr3_l_aa="QQRSNW", hla_alleles=["HLA-A*02:01"])


def test_rejects_empty_alleles() -> None:
    with pytest.raises(ValueError):
        predict_epitopes(cdr3_h_aa="ARDYYG", cdr3_l_aa="QQRSNW", hla_alleles=[])


def test_emits_runtime_error_when_mhcflurry_missing() -> None:
    if HAVE_MHCFLURRY:
        pytest.skip("mhcflurry IS installed; skipping missing-dep path test.")
    # Use a long-enough CDR3 so window enumeration produces peptides and we
    # actually hit the predictor load path.
    with pytest.raises(RuntimeError, match="MHCflurry is not installed|MHCflurry models"):
        predict_epitopes(
            cdr3_h_aa="ARDYYGSSYWYFDV",
            cdr3_l_aa="QQRSNWPPLT",
            hla_alleles=["HLA-A*02:01"],
        )


# ---------------------------------------------------------------------------
# Real prediction (slow; only when MHCflurry available)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@needs_mhcflurry
def test_predicts_epitopes_for_real_cdr3() -> None:
    out = predict_epitopes(
        cdr3_h_aa="ARDYYGSSYWYFDV",
        cdr3_l_aa="QQRSNWPPLT",
        hla_alleles=["HLA-A*02:01"],
        top_k=5,
        percentile_cutoff=10.0,           # generous cutoff so we always get hits
    )
    assert "epitopes" in out
    assert out["n_evaluated"] > 0
    # Each returned epitope must have all required fields
    for ep in out["epitopes"]:
        assert {"peptide", "hla", "affinity_nM", "percentile_rank", "length", "source_region"} <= ep.keys()
        assert ep["hla"] == "HLA-A*02:01"
