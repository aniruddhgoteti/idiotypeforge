"""Tests for the CDR liability scanner (pure regex; no external deps)."""
from __future__ import annotations

from app.tools.cdr_liabilities import run as scan_liabilities


def _vh_numbering(
    cdr1: str = "GFTFSSY",
    cdr2: str = "ISSSGGSTY",
    cdr3: str = "ARDYYGSSYWYFDV",
    # Two canonical Cys (intra-domain disulfide) so the free-cysteine check
    # doesn't fire spuriously on a healthy framework.
    framework: str = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSCAVRSAY",
) -> dict:
    """Build a minimal VH ChainNumbering dict for test inputs."""
    cdr1_start = 26
    cdr2_start = cdr1_start + len(cdr1) + 14
    cdr3_start = cdr2_start + len(cdr2) + 31
    return {
        "chain_type": "H",
        "scheme": "kabat",
        "v_gene": None,
        "j_gene": None,
        "isotype": None,
        "cdr1": {"start": cdr1_start, "end": cdr1_start + len(cdr1) - 1, "sequence": cdr1},
        "cdr2": {"start": cdr2_start, "end": cdr2_start + len(cdr2) - 1, "sequence": cdr2},
        "cdr3": {"start": cdr3_start, "end": cdr3_start + len(cdr3) - 1, "sequence": cdr3},
        "framework_sequence": framework,
    }


def _vl_numbering(
    cdr1: str = "RASQSVSSY",
    cdr2: str = "DASNRAT",
    cdr3: str = "QQRSNWPPLT",
    # Two canonical Cys (intra-domain disulfide) so the free-cysteine check
    # doesn't fire spuriously on a healthy framework.
    framework: str = "DIQMTQSPSSLSASVGDRVTITCRSGSGCSY",
) -> dict:
    cdr1_start = 24
    cdr2_start = cdr1_start + len(cdr1) + 16
    cdr3_start = cdr2_start + len(cdr2) + 31
    return {
        "chain_type": "K",
        "scheme": "kabat",
        "v_gene": None,
        "j_gene": None,
        "isotype": None,
        "cdr1": {"start": cdr1_start, "end": cdr1_start + len(cdr1) - 1, "sequence": cdr1},
        "cdr2": {"start": cdr2_start, "end": cdr2_start + len(cdr2) - 1, "sequence": cdr2},
        "cdr3": {"start": cdr3_start, "end": cdr3_start + len(cdr3) - 1, "sequence": cdr3},
        "framework_sequence": framework,
    }


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------
def test_returns_expected_keys() -> None:
    out = scan_liabilities(_vh_numbering(), _vl_numbering())
    assert set(out.keys()) == {"liabilities", "high_severity_count", "summary_by_kind"}
    assert isinstance(out["liabilities"], list)
    assert isinstance(out["high_severity_count"], int)


# ---------------------------------------------------------------------------
# N-glycosylation motif detection
# ---------------------------------------------------------------------------
def test_n_glyc_in_cdr3_is_high_severity() -> None:
    """N-X-S/T (X != P) in CDR3 must be flagged at HIGH severity."""
    vh = _vh_numbering(cdr3="ARDYNSTGFDV")    # contains NST → N-glyc
    out = scan_liabilities(vh, _vl_numbering())
    glyc_hits = [h for h in out["liabilities"] if h["kind"] == "n_glycosylation" and h["region"] == "CDR3"]
    assert len(glyc_hits) >= 1
    assert all(h["severity"] == "high" for h in glyc_hits)


def test_n_glyc_with_proline_x_is_not_flagged() -> None:
    """N-P-T does NOT match (X = Pro is the canonical exception)."""
    vh = _vh_numbering(cdr3="ARNPTYDF")        # NPT → must NOT match
    out = scan_liabilities(vh, _vl_numbering())
    glyc_hits = [h for h in out["liabilities"] if h["kind"] == "n_glycosylation"]
    # Only matches should be from the framework if any; CDR3 itself must not flag
    cdr3_glyc = [h for h in glyc_hits if h["region"] == "CDR3"]
    assert cdr3_glyc == []


# ---------------------------------------------------------------------------
# Deamidation NG
# ---------------------------------------------------------------------------
def test_ng_deamidation_motif_in_cdr3() -> None:
    vh = _vh_numbering(cdr3="ARNGDYW")
    out = scan_liabilities(vh, _vl_numbering())
    ng_hits = [h for h in out["liabilities"] if h["kind"] == "deamidation" and h["region"] == "CDR3"]
    assert any(h["motif"] == "NG" for h in ng_hits)
    # NG in CDR3 should escalate to high
    assert any(h["severity"] == "high" for h in ng_hits)


# ---------------------------------------------------------------------------
# Asp isomerization DG
# ---------------------------------------------------------------------------
def test_dg_isomerization_in_cdr2() -> None:
    vh = _vh_numbering(cdr2="ISDGYNT")
    out = scan_liabilities(vh, _vl_numbering())
    iso_hits = [h for h in out["liabilities"] if h["kind"] == "isomerization" and h["region"] == "CDR2"]
    assert any(h["motif"] == "DG" for h in iso_hits)


# ---------------------------------------------------------------------------
# Free cysteine
# ---------------------------------------------------------------------------
def test_free_cys_when_count_is_odd() -> None:
    vh = _vh_numbering(cdr3="ARCDYW")          # one Cys → odd → free
    out = scan_liabilities(vh, _vl_numbering())
    cys_hits = [h for h in out["liabilities"] if h["kind"] == "free_cysteine"]
    assert len(cys_hits) >= 1
    assert all(h["severity"] == "high" for h in cys_hits)


def test_no_free_cys_when_count_is_even() -> None:
    vh = _vh_numbering(cdr3="ARCCYW")          # two Cys → even → paired
    out = scan_liabilities(vh, _vl_numbering())
    cys_hits = [h for h in out["liabilities"] if h["kind"] == "free_cysteine" and h["region"] == "CDR3"]
    assert cys_hits == []


# ---------------------------------------------------------------------------
# Oxidation: only flagged in CDRs (not framework, to reduce noise)
# ---------------------------------------------------------------------------
def test_oxidation_flagged_in_cdr_but_not_framework() -> None:
    vh = _vh_numbering(cdr1="GFTFSWM", framework="EVQLVMSGW")  # M and W in both
    out = scan_liabilities(vh, _vl_numbering())
    cdr1_ox = [h for h in out["liabilities"] if h["kind"] == "oxidation" and h["region"] == "CDR1"]
    fr_ox = [h for h in out["liabilities"] if h["kind"] == "oxidation" and h["region"].startswith("FR")]
    assert len(cdr1_ox) >= 1
    assert fr_ox == [], "Framework oxidation should be suppressed by the scanner."


# ---------------------------------------------------------------------------
# Asp-Pro fragmentation
# ---------------------------------------------------------------------------
def test_dp_fragmentation_motif() -> None:
    vh = _vh_numbering(cdr3="ARDPYDF")
    out = scan_liabilities(vh, _vl_numbering())
    dp_hits = [h for h in out["liabilities"] if h["kind"] == "fragmentation"]
    assert any(h["motif"] == "DP" for h in dp_hits)


# ---------------------------------------------------------------------------
# Summary semantics
# ---------------------------------------------------------------------------
def test_clean_antibody_has_low_high_severity_count() -> None:
    """A 'clean' antibody (no NXS/T, no NG, no DG, even Cys, no DP in CDRs)
    should produce a small high-severity count (oxidation in CDRs is OK)."""
    vh = _vh_numbering(
        cdr1="GFTFSSY",
        cdr2="ISSSGGSTY",
        cdr3="ARYYGSSYYFDV",          # no NXS/T, no NG, no DG
        # default framework has both canonical Cys
    )
    out = scan_liabilities(vh, _vl_numbering(cdr3="QQYYSPLT"))
    # Some oxidation hits are fine (W, M); none should be high-severity glyc/NG/free-Cys/iso.
    high_kinds = {h["kind"] for h in out["liabilities"] if h["severity"] == "high"}
    assert "n_glycosylation" not in high_kinds
    assert "free_cysteine" not in high_kinds


def test_summary_by_kind_counts_correctly() -> None:
    vh = _vh_numbering(cdr3="ARNGSDGCYW")     # NG + DG + 1 free Cys
    out = scan_liabilities(vh, _vl_numbering())
    s = out["summary_by_kind"]
    assert s.get("deamidation", 0) >= 1
    assert s.get("isomerization", 0) >= 1
    assert s.get("free_cysteine", 0) >= 1
