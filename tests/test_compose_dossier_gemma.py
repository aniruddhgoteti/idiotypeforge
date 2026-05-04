"""Tests for the Gemma-mode dossier composer.

The Gemma path always returns a complete dossier — the banner-and-fallback
behaviour means *no* configuration of Ollama can make the function raise.
These tests stub `ollama.chat` to exercise each branch:

  - ImportError      → banner + template fallback
  - chat raises      → banner + template fallback
  - empty content    → banner + template fallback
  - happy path       → Gemma's markdown returned verbatim, citations parsed

`_extract_citations` is also unit-tested against the same regex
CitationGate uses, so a string that passes here will pass there.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from app.tools import compose_dossier


# ---------------------------------------------------------------------------
# Minimal artifact fixture (matches the shape _run_template builds)
# ---------------------------------------------------------------------------
@pytest.fixture
def artifacts() -> dict[str, Any]:
    return {
        "patient_id": "pt_test",
        "bcr_summary": {
            "vh_v_gene": "IGHV4-34",
            "vh_j_gene": "IGHJ4",
            "vh_cdr3": "ARDRGYYFDY",
            "vl_cdr3": "QQYNSYPLT",
        },
        "top_mrna_peptides": [
            {
                "peptide": "ARDRGYYFDY",
                "length": 10,
                "hla": "HLA-A*02:01",
                "affinity_nM": 42.0,
                "percentile_rank": 0.5,
                "source_region": "CDR3-H",
            }
        ],
        "top_binders": [
            {
                "candidate_id": "C1",
                "sequence": "AAA",
                "iplddt": 0.85,
                "ipae": 9.1,
                "interface_sasa": 1100.0,
                "proteinmpnn_logprob": -1.2,
                "calibrated_p_binder": 0.7,
            }
        ],
        "car_construct": {
            "format": "4-1BBz",
            "full_aa_sequence": "M" * 480,
            "components": {},
        },
        "off_target_report": {
            "max_identity_pct": 35.0,
            "n_hits_above_70pct": 0,
        },
    }


# ---------------------------------------------------------------------------
# _extract_citations — pure regex helper
# ---------------------------------------------------------------------------
def test_extract_citations_dedupes_and_orders() -> None:
    md = (
        "Refs include [Maude2018], [Schuster2011], and [Maude2018] again. "
        "Multi: [Watson2023, Dauparas2022]. Year-only [2024] is ignored."
    )
    keys = compose_dossier._extract_citations(md)
    assert keys == ["Maude2018", "Schuster2011", "Watson2023", "Dauparas2022"]


def test_extract_citations_empty_when_no_brackets() -> None:
    assert compose_dossier._extract_citations("just prose, no citations") == []


# ---------------------------------------------------------------------------
# ImportError path — ollama package missing
# ---------------------------------------------------------------------------
def test_gemma_falls_back_when_ollama_missing(
    monkeypatch: pytest.MonkeyPatch, artifacts: dict[str, Any]
) -> None:
    # Make `import ollama` raise inside _compose_with_gemma without affecting
    # the real environment by injecting a sentinel that raises on import.
    monkeypatch.setitem(sys.modules, "ollama", None)
    monkeypatch.setenv("IDIOTYPEFORGE_DOSSIER_MODE", "gemma")

    out = compose_dossier.run(**artifacts)

    assert out["mode"] == "template_fallback_for_gemma"
    assert "Gemma 4 unavailable" in out["markdown"]
    assert "## 1. BCR fingerprint" in out["markdown"]


# ---------------------------------------------------------------------------
# Exception path — ollama.chat raises
# ---------------------------------------------------------------------------
def test_gemma_falls_back_when_chat_raises(
    monkeypatch: pytest.MonkeyPatch, artifacts: dict[str, Any]
) -> None:
    fake = types.ModuleType("ollama")

    def boom(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise ConnectionError("daemon not running on 127.0.0.1:11434")

    fake.chat = boom            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake)
    monkeypatch.setenv("IDIOTYPEFORGE_DOSSIER_MODE", "gemma")

    out = compose_dossier.run(**artifacts)

    assert out["mode"] == "template_fallback_for_gemma"
    assert "Gemma 4 generation failed" in out["markdown"]
    assert "ConnectionError" in out["markdown"]
    assert "## 1. BCR fingerprint" in out["markdown"]


# ---------------------------------------------------------------------------
# Empty-content path — Ollama returns "" (model produced nothing)
# ---------------------------------------------------------------------------
def test_gemma_falls_back_on_empty_content(
    monkeypatch: pytest.MonkeyPatch, artifacts: dict[str, Any]
) -> None:
    fake = types.ModuleType("ollama")
    fake.chat = lambda **_kw: {"message": {"content": "   \n"}}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake)
    monkeypatch.setenv("IDIOTYPEFORGE_DOSSIER_MODE", "gemma")

    out = compose_dossier.run(**artifacts)

    assert out["mode"] == "template_fallback_for_gemma"
    assert "empty dossier" in out["markdown"]


# ---------------------------------------------------------------------------
# Happy path — Gemma returns a usable markdown
# ---------------------------------------------------------------------------
def test_gemma_happy_path_returns_gemma_mode(
    monkeypatch: pytest.MonkeyPatch, artifacts: dict[str, Any]
) -> None:
    gemma_md = (
        "# Personalized Therapy Dossier · pt_test\n"
        "Pioneered by [Schuster2011]; CD19 CAR-T evidence in [Maude2018].\n"
        "Designed-binder rationale draws on [Watson2023, Dauparas2022].\n"
    )
    fake = types.ModuleType("ollama")
    fake.chat = lambda **_kw: {"message": {"content": gemma_md}}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake)
    monkeypatch.setenv("IDIOTYPEFORGE_DOSSIER_MODE", "gemma")

    out = compose_dossier.run(**artifacts)

    assert out["mode"] == "gemma"
    # The composer strips trailing whitespace; otherwise content is verbatim.
    assert out["markdown"] == gemma_md.strip()
    assert out["citations"] == ["Schuster2011", "Maude2018", "Watson2023", "Dauparas2022"]


# ---------------------------------------------------------------------------
# Default mode is unchanged — IDIOTYPEFORGE_DOSSIER_MODE unset
# ---------------------------------------------------------------------------
def test_default_mode_is_template(
    monkeypatch: pytest.MonkeyPatch, artifacts: dict[str, Any]
) -> None:
    monkeypatch.delenv("IDIOTYPEFORGE_DOSSIER_MODE", raising=False)
    out = compose_dossier.run(**artifacts)
    assert out["mode"] == "template"
    # No banner — the template path is the default and shouldn't be flagged.
    assert "Gemma 4" not in out["markdown"].split("##", 1)[0]
