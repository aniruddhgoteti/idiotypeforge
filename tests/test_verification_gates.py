"""Tests for the verification gate framework.

Anti-hallucination is the most important guarantee this project makes;
these tests are the safety net for that guarantee.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.verification import (
    ArtifactStore,
    CitationGate,
    GateRunner,
    MockModeGate,
    ProvenanceGate,
    SchemaGate,
    ThresholdGate,
)
from app.verification.gates import Threshold
from app.verification.provenance import numeric_aliases, walk_numbers


# ===========================================================================
# Provenance & ArtifactStore
# ===========================================================================
def test_numeric_aliases_covers_common_formats() -> None:
    aliases = numeric_aliases(0.873)
    # the model might write any of these:
    assert "0.873" in aliases
    assert "0.87" in aliases
    # percentage form (since 0 ≤ 0.873 ≤ 1)
    assert "87.3" in aliases or "87" in aliases


def test_walk_numbers_recurses_into_nested_structures() -> None:
    payload = {"a": 1.5, "b": {"c": [2, 3, {"d": 4.0}]}}
    pairs = list(walk_numbers(payload))
    values = [v for v, _ in pairs]
    paths = [p for _, p in pairs]
    assert 1.5 in values
    assert 4.0 in values
    assert "$.b.c[2].d" in paths


def test_artifact_store_records_call_indices() -> None:
    store = ArtifactStore()
    store.record("igfold_predict", {"vh": "EVQ"}, {"mean_plddt": 0.86})
    store.record("igfold_predict", {"vh": "DIQ"}, {"mean_plddt": 0.91})
    assert store.artifacts[0].artifact_id == "igfold_predict:0"
    assert store.artifacts[1].artifact_id == "igfold_predict:1"


def test_artifact_store_indexes_aliases_globally() -> None:
    store = ArtifactStore()
    store.record("rescore_complex", {}, {"iplddt": 0.847, "ipae": 6.2})
    assert store.has_alias("0.85")     # 2-decimal alias of 0.847
    assert store.has_alias("6.2")       # exact
    hits = store.lookup("0.85")
    assert len(hits) == 1
    assert hits[0][0] == "rescore_complex:0"


def test_artifact_store_flags_mocks() -> None:
    store = ArtifactStore()
    store.record("design_binder", {}, {"candidates": [], "mock": True})
    store.record("number_antibody", {}, {"vh": {}})
    mocks = store.mock_artifacts()
    assert len(mocks) == 1
    assert mocks[0].tool_name == "design_binder"


# ===========================================================================
# SchemaGate
# ===========================================================================
def test_schema_gate_accepts_valid_payload() -> None:
    from app.tools._types import CARConstruct
    payload = {
        "format": "4-1BBz",
        "full_aa_sequence": "M" * 500,
        "components": {"x": "y"},
    }
    g = SchemaGate(CARConstruct)
    r = g.check(output=payload)
    assert r.passed


def test_schema_gate_rejects_invalid_payload() -> None:
    from app.tools._types import CARConstruct
    g = SchemaGate(CARConstruct)
    r = g.check(output={"format": "WrongFormat", "components": {}})
    assert not r.passed
    assert r.severity == "error"


# ===========================================================================
# MockModeGate
# ===========================================================================
def test_mock_mode_gate_dev_mode_passes_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    store = ArtifactStore()
    store.record("design_binder", {}, {"candidates": [], "mock": True})
    r = MockModeGate().check(store=store)
    assert r.passed
    assert r.severity == "info"


def test_mock_mode_gate_prod_mode_fails_critical_on_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "0")
    store = ArtifactStore()
    store.record("design_binder", {}, {"candidates": [], "mock": True})
    r = MockModeGate().check(store=store)
    assert not r.passed
    assert r.severity == "critical"


def test_mock_mode_gate_passes_with_no_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "0")
    store = ArtifactStore()
    store.record("number_antibody", {}, {"vh": {"chain_type": "H"}})
    r = MockModeGate().check(store=store)
    assert r.passed


# ===========================================================================
# ThresholdGate
# ===========================================================================
def test_threshold_gate_passes_when_within_bounds() -> None:
    output = {
        "iplddt": 0.85, "ipae": 6.2, "interface_sasa": 950.0,
        "max_identity_pct": 35.0,
        "mean_plddt": 0.88, "cdr3_mean_plddt": 0.74,
    }
    r = ThresholdGate().check(output=output)
    assert r.passed
    # No warnings expected
    assert r.severity == "info"


def test_threshold_gate_fails_on_low_iplddt() -> None:
    output = {"iplddt": 0.40, "ipae": 8.0, "interface_sasa": 800.0,
              "max_identity_pct": 25.0, "mean_plddt": 0.85, "cdr3_mean_plddt": 0.72}
    r = ThresholdGate().check(output=output)
    assert not r.passed
    assert r.severity == "error"
    failure_names = [f["name"] for f in r.details["failures"]]
    assert "rescore.iplddt" in failure_names


def test_threshold_gate_warns_on_borderline_values() -> None:
    """ipLDDT 0.65 passes the hard 0.50 floor but breaches the soft 0.70 warn."""
    output = {"iplddt": 0.65, "ipae": 8.0, "interface_sasa": 800.0,
              "max_identity_pct": 25.0, "mean_plddt": 0.85, "cdr3_mean_plddt": 0.72}
    r = ThresholdGate().check(output=output)
    assert r.passed                                # no hard failures
    assert r.severity == "warning"


def test_threshold_gate_fails_on_high_offtarget_identity() -> None:
    output = {"iplddt": 0.85, "ipae": 6.0, "interface_sasa": 900.0,
              "max_identity_pct": 78.0,           # > 70 % → fail
              "mean_plddt": 0.86, "cdr3_mean_plddt": 0.7}
    r = ThresholdGate().check(output=output)
    assert not r.passed
    failure_names = [f["name"] for f in r.details["failures"]]
    assert "offtarget.max_identity" in failure_names


def test_threshold_gate_supports_subset_filter() -> None:
    output = {"mean_plddt": 0.50}
    r = ThresholdGate().check(output=output, threshold_subset=["igfold.mean_plddt"])
    assert not r.passed


def test_threshold_gate_resolves_nested_paths() -> None:
    output = {"foo": {"iplddt": 0.85, "ipae": 7.0, "interface_sasa": 900.0,
                       "max_identity_pct": 30.0, "mean_plddt": 0.85, "cdr3_mean_plddt": 0.7}}
    custom = [Threshold("nested.iplddt", "$.foo.iplddt", ">=", 0.7)]
    r = ThresholdGate(thresholds=custom).check(output=output)
    assert r.passed


# ===========================================================================
# ProvenanceGate — anti-hallucination
# ===========================================================================
def test_provenance_gate_passes_when_all_numbers_traceable() -> None:
    store = ArtifactStore()
    store.record(
        "rescore_complex",
        {},
        {"iplddt": 0.847, "ipae": 6.2, "interface_sasa": 942.0, "max_identity_pct": 32.5},
    )
    dossier = (
        "Top binder: ipLDDT 0.85, iPAE 6.2 Å, interface SASA 942 Å². "
        "Off-target maximum identity 32.5%."
    )
    r = ProvenanceGate().check(dossier_markdown=dossier, store=store)
    assert r.passed, r.reasons


def test_provenance_gate_catches_hallucinated_value() -> None:
    """Gemma 4 invents a number that no tool ever returned."""
    store = ArtifactStore()
    store.record("rescore_complex", {}, {"iplddt": 0.847})
    dossier = "The agent reported an ipLDDT of 0.999 — but no tool produced this."
    r = ProvenanceGate().check(dossier_markdown=dossier, store=store)
    assert not r.passed
    assert r.severity == "critical"
    assert any("not traceable" in reason for reason in r.reasons)


def test_provenance_gate_ignores_years_and_small_integers() -> None:
    store = ArtifactStore()
    store.record("number_antibody", {}, {"vh": {"chain_type": "H"}})
    dossier = (
        "BiovaxID failed in 2011. The antibody has 3 CDR loops on each chain. "
        "Patient enrolled in 2024 cohort 1."
    )
    r = ProvenanceGate().check(dossier_markdown=dossier, store=store)
    assert r.passed


def test_provenance_gate_tolerates_max_unmatched() -> None:
    store = ArtifactStore()
    store.record("rescore_complex", {}, {"iplddt": 0.85})
    dossier = "ipLDDT 0.85, plus an unrelated 88.7 from the LLM's training data."
    # max_unmatched=0 → fails
    assert not ProvenanceGate(max_unmatched=0).check(dossier_markdown=dossier, store=store).passed
    # max_unmatched=1 → passes
    assert ProvenanceGate(max_unmatched=1).check(dossier_markdown=dossier, store=store).passed


def test_provenance_gate_handles_percent_unit_aliases() -> None:
    """If the tool returned 0.325 (a fraction), the dossier might write 32.5% or 0.325 — both should resolve."""
    store = ArtifactStore()
    store.record("offtarget_search", {}, {"max_identity": 0.325})
    dossier_a = "Off-target identity 32.5%."
    dossier_b = "Off-target identity 0.325."
    assert ProvenanceGate().check(dossier_markdown=dossier_a, store=store).passed
    assert ProvenanceGate().check(dossier_markdown=dossier_b, store=store).passed


# ===========================================================================
# CitationGate
# ===========================================================================
def test_citation_gate_passes_with_known_keys(tmp_path: Path) -> None:
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{Schuster2011, title={x}}\n@article{Maude2018, title={y}}\n")
    dossier = "See [Schuster2011] and [Maude2018] for the trial designs."
    r = CitationGate(bib_path=bib).check(dossier_markdown=dossier)
    assert r.passed


def test_citation_gate_catches_invented_citation(tmp_path: Path) -> None:
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{Schuster2011, title={x}}\n")
    dossier = "Per [Schuster2011] and the speculative [HallucinatedAuthor2025]."
    r = CitationGate(bib_path=bib).check(dossier_markdown=dossier)
    assert not r.passed
    assert "HallucinatedAuthor2025" in r.details["unknown_keys"]


def test_citation_gate_handles_real_references_bib() -> None:
    """Run against the actual project bibliography."""
    real_bib = Path("data/references.bib")
    if not real_bib.exists():
        pytest.skip("data/references.bib not found in this run")
    dossier = "Idiotype vaccine history per [Schuster2011] and ANARCI per [Dunbar2016]."
    r = CitationGate(bib_path=real_bib).check(dossier_markdown=dossier)
    assert r.passed, r.details


# ===========================================================================
# Runner
# ===========================================================================
def test_runner_aborts_on_first_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "0")
    store = ArtifactStore()
    store.record("design_binder", {}, {"candidates": [], "mock": True})
    runner = GateRunner(abort_on="error")
    overall, results = runner.run([
        (MockModeGate(), {"store": store}),
        (ThresholdGate(), {"output": {}}),     # would normally pass; should be skipped
    ])
    assert not overall
    # Aborted after MockModeGate (critical fail)
    assert len(results) == 1
    assert results[0].gate_name == "MockModeGate"


def test_runner_continues_past_warnings() -> None:
    runner = GateRunner(abort_on="error")
    output = {"iplddt": 0.65, "ipae": 8.0, "interface_sasa": 800.0,
              "max_identity_pct": 25.0, "mean_plddt": 0.85, "cdr3_mean_plddt": 0.72}
    overall, results = runner.run([
        (ThresholdGate(), {"output": output}),
    ])
    assert overall
    assert results[0].severity == "warning"


def test_runner_report_markdown_is_human_readable() -> None:
    runner = GateRunner()
    store = ArtifactStore()
    store.record("rescore_complex", {}, {"iplddt": 0.85})
    _, results = runner.run([(MockModeGate(), {"store": store})])
    md = runner.report_markdown(results)
    assert "## Verification audit" in md
    assert "MockModeGate" in md
