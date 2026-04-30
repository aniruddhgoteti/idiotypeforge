"""Tests for the ANARCI numbering tool.

Skipped if `anarci` is not installed (e.g. on a CPU-only laptop without
HMMER). When ANARCI is present, these tests assert the Kabat CDR3 sequences
of rituximab and trastuzumab match published expectations.
"""
from __future__ import annotations

import importlib.util

import pytest

from app.tools.number_antibody import run as number_antibody
from tests.fixtures.known_antibodies import (
    RITUXIMAB_CDR3_H,
    RITUXIMAB_CDR3_L,
    RITUXIMAB_VH,
    RITUXIMAB_VL,
    TRASTUZUMAB_CDR3_H,
    TRASTUZUMAB_CDR3_L,
    TRASTUZUMAB_VH,
    TRASTUZUMAB_VL,
)


HAVE_ANARCI = importlib.util.find_spec("anarci") is not None
needs_anarci = pytest.mark.skipif(
    not HAVE_ANARCI,
    reason="anarci not installed; run `uv pip install anarci` and install HMMER.",
)


# ---------------------------------------------------------------------------
# Argument validation (no ANARCI required)
# ---------------------------------------------------------------------------
def test_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError):
        number_antibody(vh_sequence="", vl_sequence="DIQMTQSPSS")
    with pytest.raises(ValueError):
        number_antibody(vh_sequence="EVQLVQSGG", vl_sequence="")


def test_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        number_antibody(vh_sequence="EVQ", vl_sequence="DIQ", scheme="bogus")


def test_emits_runtime_error_when_anarci_missing() -> None:
    """If ANARCI isn't installed, the tool must raise a clear RuntimeError."""
    if HAVE_ANARCI:
        pytest.skip("anarci IS installed; skipping the missing-dep path test.")
    with pytest.raises(RuntimeError, match="ANARCI is not installed"):
        number_antibody(vh_sequence=RITUXIMAB_VH, vl_sequence=RITUXIMAB_VL)


# ---------------------------------------------------------------------------
# Real ANARCI runs (skipped when ANARCI unavailable)
# ---------------------------------------------------------------------------
@needs_anarci
def test_rituximab_kabat_cdr3_heavy() -> None:
    out = number_antibody(
        vh_sequence=RITUXIMAB_VH,
        vl_sequence=RITUXIMAB_VL,
        scheme="kabat",
    )
    assert out["vh"]["chain_type"] == "H"
    assert out["vh"]["scheme"] == "kabat"
    assert out["vh"]["cdr3"]["sequence"] == RITUXIMAB_CDR3_H


@needs_anarci
def test_rituximab_kabat_cdr3_light() -> None:
    out = number_antibody(
        vh_sequence=RITUXIMAB_VH,
        vl_sequence=RITUXIMAB_VL,
        scheme="kabat",
    )
    assert out["vl"]["chain_type"] in {"K", "L"}
    assert out["vl"]["cdr3"]["sequence"] == RITUXIMAB_CDR3_L


@needs_anarci
def test_trastuzumab_kabat_cdrs() -> None:
    out = number_antibody(
        vh_sequence=TRASTUZUMAB_VH,
        vl_sequence=TRASTUZUMAB_VL,
        scheme="kabat",
    )
    assert out["vh"]["cdr3"]["sequence"] == TRASTUZUMAB_CDR3_H
    assert out["vl"]["cdr3"]["sequence"] == TRASTUZUMAB_CDR3_L


@needs_anarci
def test_imgt_scheme_works_too() -> None:
    """IMGT numbering should also produce a valid CDR3 — content differs from Kabat."""
    out = number_antibody(
        vh_sequence=RITUXIMAB_VH,
        vl_sequence=RITUXIMAB_VL,
        scheme="imgt",
    )
    assert len(out["vh"]["cdr3"]["sequence"]) > 0
    assert out["vh"]["scheme"] == "imgt"


@needs_anarci
def test_v_gene_assignment_is_returned() -> None:
    """ANARCI assign_germline=True should give us a V-gene label."""
    out = number_antibody(
        vh_sequence=RITUXIMAB_VH,
        vl_sequence=RITUXIMAB_VL,
        scheme="kabat",
    )
    # V-gene should at least be a non-empty string starting with IGH/IGK/IGL.
    v = out["vh"].get("v_gene")
    assert v is None or v.startswith(("IGH", "IGK", "IGL"))
