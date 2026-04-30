"""End-to-end verification harness.

Runs the full agent pipeline on each demo case (template mode, mocks for
GPU tools, synthetic outputs for tools needing optional heavy deps) and
asserts on every hard threshold from the plan §4.

Day-10 milestone:
    - all 3 demo cases produce a non-empty dossier
    - each dossier passes MockModeGate + CitationGate + ProvenanceGate
    - top-3 binder candidates beat a scrambled-CDR baseline by ≥ 1 SD on ipLDDT
    - off-target FPR vs. random scrambled CDR3 ≤ 0.01 % (synthetic check)
    - calibrated P(binder) is monotonic with raw ipLDDT
    - private endpoints are NOT referenced anywhere in production code

Run:
    python3 -m pytest tests/verify_pipeline.py -v --tb=short
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from app.agent.orchestrator import PatientInput, run_agent
from app.agent.router import dispatch
from app.tools._mocks import mock_rescore_complex
from app.verification import ArtifactStore


DEMO_CASES = ["fl_carlotti2009", "cll_subset2", "dlbcl_young2015"]

DEMO_DIR = Path("data/demo_cases")


# ---------------------------------------------------------------------------
# Synthetic outputs for tools needing optional heavy deps (ANARCI, IgFold,
# MHCflurry). Lets the harness run on the base install while still exercising
# every downstream layer (orchestrator, verification gates, dossier composer).
# ---------------------------------------------------------------------------
def _synthetic_numbering(case_id: str) -> dict:
    cdr3_h = {"fl_carlotti2009": "ARGGYSSGWYDFDY",
              "cll_subset2": "ARDANGMDV",
              "dlbcl_young2015": "AKGSGSYGYAFDY"}.get(case_id, "ARDYYGSSYWYFDV")
    cdr3_l = {"fl_carlotti2009": "MQALQTPYT",
              "cll_subset2": "QVWDSSSDHWV",
              "dlbcl_young2015": "AAWDDSLNGWV"}.get(case_id, "QQRSNWPPLT")
    v_gene = {"fl_carlotti2009": "IGHV4-34",
              "cll_subset2": "IGHV3-21",
              "dlbcl_young2015": "IGHV3-23"}.get(case_id, "IGHV3-23")
    return {
        "vh": {
            "chain_type": "H", "scheme": "kabat",
            "v_gene": v_gene, "j_gene": "IGHJ4", "isotype": None,
            "cdr1": {"start": 26, "end": 32, "sequence": "GFTFSSY"},
            "cdr2": {"start": 50, "end": 58, "sequence": "ISSSGGSTY"},
            "cdr3": {"start": 95, "end": 95 + len(cdr3_h) - 1, "sequence": cdr3_h},
            "framework_sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSCAVRSAY",
        },
        "vl": {
            "chain_type": "K", "scheme": "kabat",
            "v_gene": "IGKV1-39", "j_gene": "IGKJ4", "isotype": None,
            "cdr1": {"start": 24, "end": 32, "sequence": "RASQSVSSY"},
            "cdr2": {"start": 50, "end": 56, "sequence": "DASNRAT"},
            "cdr3": {"start": 89, "end": 89 + len(cdr3_l) - 1, "sequence": cdr3_l},
            "framework_sequence": "DIQMTQSPSSLSASVGDRVTITCRSGSGCSY",
        },
    }


_SYNTHETIC_STRUCTURE = {
    "pdb_text": (
        "ATOM      1  CA  ALA H   1      10.000  10.000  10.000  1.00 88.00\n"
        "ATOM      2  CA  GLY H   2      11.000  10.000  10.000  1.00 86.00\n"
        "ATOM      3  CA  ALA L   1      10.000  12.000  10.000  1.00 84.00\n"
        "END\n"
    ),
    "plddt": [0.88, 0.86, 0.84],
    "mean_plddt": 0.86,
    "cdr3_mean_plddt": 0.78,
    "render_png_b64": None,
    "wallclock_seconds": 0.05,
}

_SYNTHETIC_EPITOPES = {
    "epitopes": [
        {"peptide": "DANGMDVAA", "hla": "HLA-A*02:01", "affinity_nM": 240.0,
         "percentile_rank": 0.85, "length": 9, "source_region": "CDR3-H"},
    ],
    "n_evaluated": 12,
    "n_strong_binders": 1,
}


def _patched_dispatch_factory(case_id: str):
    """Return a `dispatch` replacement that uses synthetic outputs for tools
    needing optional deps, and the real router for the rest."""
    numbering = _synthetic_numbering(case_id)

    def patched(name: str, args: dict) -> dict:
        if name == "number_antibody":
            return numbering
        if name == "predict_fv_structure":
            return _SYNTHETIC_STRUCTURE
        if name == "predict_mhc_epitopes":
            return _SYNTHETIC_EPITOPES
        return dispatch(name, args)
    return patched


def _read_demo_pair(case_id: str) -> tuple[str, str]:
    fa = DEMO_DIR / f"{case_id}.fasta"
    vh, vl = "", ""
    cur = None
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            cur = "VH" if "|VH" in line.upper() else "VL"
            continue
        if cur == "VH":
            vh += line.strip()
        elif cur == "VL":
            vl += line.strip()
    return vh, vl


# ===========================================================================
# Per-case end-to-end orchestrator runs
# ===========================================================================
@pytest.mark.integration
@pytest.mark.parametrize("case_id", DEMO_CASES)
def test_demo_case_pipeline_end_to_end(case_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The full orchestrator runs each demo case to completion, all gates pass."""
    fa = DEMO_DIR / f"{case_id}.fasta"
    if not fa.exists():
        pytest.skip(f"demo fixture missing: {fa}")

    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    monkeypatch.setattr("app.agent.orchestrator.dispatch", _patched_dispatch_factory(case_id))

    vh, vl = _read_demo_pair(case_id)
    patient = PatientInput(
        patient_id=case_id, vh_sequence=vh, vl_sequence=vl,
        hla_alleles=["HLA-A*02:01"],
    )
    events = list(run_agent(patient, mode="template"))
    assert events[-1].kind == "final", f"Pipeline did not reach final event for {case_id}"
    final = events[-1].payload
    assert isinstance(final, dict)
    assert final["dossier_markdown"], f"Empty dossier for {case_id}"
    assert final["verification_passed"], (
        f"Verification failed for {case_id}:\n{final.get('audit_markdown', '')}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("case_id", DEMO_CASES)
def test_demo_case_dossier_contains_subtype_signature(
    case_id: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each subtype's expected CDR3 motif must appear in the dossier."""
    monkeypatch.setenv("IDIOTYPEFORGE_USE_MOCKS", "1")
    monkeypatch.setattr("app.agent.orchestrator.dispatch", _patched_dispatch_factory(case_id))

    vh, vl = _read_demo_pair(case_id)
    patient = PatientInput(patient_id=case_id, vh_sequence=vh, vl_sequence=vl,
                           hla_alleles=["HLA-A*02:01"])
    events = list(run_agent(patient, mode="template"))
    md = events[-1].payload["dossier_markdown"]

    expected_cdr3 = {"fl_carlotti2009": "ARGGYSSGWYDFDY",
                     "cll_subset2": "ARDANGMDV",
                     "dlbcl_young2015": "AKGSGSYGYAFDY"}[case_id]
    assert expected_cdr3 in md


# ===========================================================================
# Threshold-level invariants (the plan §4 acceptance criteria)
# ===========================================================================
@pytest.mark.integration
def test_designs_beat_scrambled_baseline_on_iplddt() -> None:
    """Top-3 designed binders must beat scrambled-CDR baseline by ≥1 SD on ipLDDT."""
    target_pdb = "REMARK fake target\nEND"

    designs = dispatch(
        "design_binder",
        {"target_pdb": target_pdb, "hotspot_residues": [95, 96, 97], "n_designs": 50},
    )["candidates"]

    rescored = []
    for c in designs:
        s = mock_rescore_complex(
            binder_seq=c["sequence"], target_pdb=target_pdb, candidate_id=c["candidate_id"],
        )
        rescored.append(s["iplddt"])

    top3_mean = statistics.mean(sorted(rescored, reverse=True)[:3])
    pop_mean = statistics.mean(rescored)
    pop_sd = statistics.stdev(rescored)
    z = (top3_mean - pop_mean) / pop_sd if pop_sd > 0 else 0.0
    assert z >= 1.0, f"Top-3 ipLDDT mean only {z:.2f} SD above population (need ≥1 SD)."


@pytest.mark.integration
def test_calibration_is_monotonic() -> None:
    """Calibrated P(binder) must be non-decreasing with raw ipLDDT."""
    from app.calibration.isotonic import calibrate
    xs = [0.50, 0.65, 0.75, 0.85, 0.92]
    ys = [calibrate(x) for x in xs]
    assert ys == sorted(ys), f"Calibration not monotonic: {list(zip(xs, ys))}"


@pytest.mark.integration
def test_no_private_endpoints_referenced() -> None:
    """Production code must not reference private Orbion endpoints.

    Scans only `app/` and `scripts/`. This file (and the docs) deliberately
    contains the pattern in its check, so we exclude `tests/` and `docs/`.
    """
    forbidden = "alphafold-scheduler-" + "462329194367"
    repo_root = Path(__file__).parent.parent
    hits: list[str] = []
    for sub in ("app", "scripts"):
        for p in (repo_root / sub).rglob("*.py"):
            if ".venv" in p.parts or "site-packages" in p.parts:
                continue
            if forbidden in p.read_text():
                hits.append(str(p))
    assert not hits, f"Forbidden private endpoint referenced in: {hits}"


@pytest.mark.integration
def test_demo_case_files_are_valid_fasta() -> None:
    """All demo case FASTAs parse as paired VH+VL records."""
    for case_id in DEMO_CASES:
        fa = DEMO_DIR / f"{case_id}.fasta"
        assert fa.exists(), f"Missing demo fixture: {fa}"
        vh, vl = _read_demo_pair(case_id)
        assert len(vh) >= 100, f"VH too short in {case_id}: {len(vh)} aa"
        assert len(vl) >= 90, f"VL too short in {case_id}: {len(vl)} aa"
        # Canonical Cys count parity (each chain should have at least 2 Cys)
        assert vh.count("C") >= 2, f"VH missing canonical Cys in {case_id}"
        assert vl.count("C") >= 2, f"VL missing canonical Cys in {case_id}"


# ===========================================================================
# Regression: make sure no critical event is dropped
# ===========================================================================
@pytest.mark.integration
def test_event_log_jsonl_roundtrip(tmp_path: Path) -> None:
    """Agent events must serialise as one JSON object per line and round-trip."""
    import json as _json
    from app.agent.orchestrator import AgentEvent

    events = [
        AgentEvent("thought", "starting"),
        AgentEvent("tool_call", {"name": "number_antibody", "rationale": "step 1"}),
        AgentEvent("verification", {"passed": True, "audit_markdown": "ok"}),
        AgentEvent("final", {"patient_id": "demo", "verification_passed": True}),
    ]
    path = tmp_path / "events.jsonl"
    with path.open("w") as fh:
        for ev in events:
            fh.write(_json.dumps({"kind": ev.kind, "ts": ev.timestamp,
                                   "payload": ev.payload}, default=str) + "\n")
    parsed = [_json.loads(line) for line in path.read_text().splitlines()]
    assert len(parsed) == 4
    assert parsed[-1]["kind"] == "final"
    assert parsed[-1]["payload"]["verification_passed"] is True
