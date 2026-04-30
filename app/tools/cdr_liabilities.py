"""CDR liability scanner.

Scans CDR loops + framework for known developability liabilities:
  - N-glycosylation motif (N-X-S/T, X != P)
  - Asn deamidation hotspots (NG, NS)
  - Asp isomerization (DG, DS, DT, DD, DH)
  - Met / Trp oxidation
  - Free cysteines (unpaired Cys)
  - Asp-Pro fragmentation (DP)

Pure-python regex over the numbered chain. CPU-only, zero external deps.

References:
  Jain et al. 2017 PNAS — biophysical liability profiling of clinical-stage mAbs
  Raybould et al. 2019 PNAS — Therapeutic Antibody Profiler
"""
from __future__ import annotations

import re
from typing import Any

from ._types import ChainNumbering, Liability


SCHEMA = {
    "name": "score_cdr_liabilities",
    "description": (
        "Scan CDR + framework sequences for developability liabilities (N-glyc, "
        "deamidation, isomerization, oxidation, free Cys, Asp-Pro). Returns a "
        "list of liabilities with severity ratings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "vh_numbering": {"type": "object"},
            "vl_numbering": {"type": "object"},
        },
        "required": ["vh_numbering", "vl_numbering"],
    },
}


# Regions weighted by impact: a CDR3 hotspot is more severe than a buried framework one.
_REGION_BASE_SEVERITY: dict[str, str] = {
    "FR1": "low", "FR2": "low", "FR3": "low", "FR4": "low",
    "CDR1": "medium", "CDR2": "medium", "CDR3": "high",
}

# Per-motif baseline severity. The final severity is max(motif_base, region_base).
_MOTIF_BASE_SEVERITY: dict[str, str] = {
    "n_glycosylation": "high",   # disrupts effector function, complicates manufacturing
    "deamidation":     "medium",
    "isomerization":   "medium",
    "oxidation":       "low",    # M/W are common; only flagged for visibility
    "free_cysteine":   "high",   # always an issue
    "fragmentation":   "medium",
}

_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def _max_severity(a: str, b: str) -> str:
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


# ---------------------------------------------------------------------------
# Motif regexes (compile once)
# Notes on syntax:
#   N[^P][ST]   classic eukaryotic N-glyc consensus (N-X-S/T, X != P)
#   N[GS]       deamidation hotspots (NG most severe; NS milder, included)
#   D[GSTDH]    Asp isomerization motifs (DG most severe)
#   [MW]        oxidation hotspots — Met and Trp
#   C           any cysteine; we count Cys-pair parity later
#   DP          fragmentation (acid-labile)
# ---------------------------------------------------------------------------
_MOTIFS: dict[str, re.Pattern[str]] = {
    "n_glycosylation": re.compile(r"N[^P][ST]"),
    "deamidation":     re.compile(r"N[GS]"),
    "isomerization":   re.compile(r"D[GSTDH]"),
    "oxidation":       re.compile(r"[MW]"),
    "fragmentation":   re.compile(r"DP"),
}


def _scan_chain(numbering: ChainNumbering) -> list[Liability]:
    """Walk every region of a chain and collect liability hits."""
    out: list[Liability] = []

    # Build region segments with their (region_name, start_pos, sequence) tuples.
    segments: list[tuple[str, int, str]] = []

    # Framework1 = everything before CDR1
    full = numbering.framework_sequence
    cdr1, cdr2, cdr3 = numbering.cdr1, numbering.cdr2, numbering.cdr3

    # We don't have explicit FR boundaries from the typed model; reconstruct
    # them from CDR spans. The framework_sequence here is the *concatenated*
    # framework regions (FR1+FR2+FR3+FR4) from ANARCI's _parse helper.
    # For the regex scan, we treat each CDR + a single combined "FR" region.
    segments.append(("CDR1", cdr1.start, cdr1.sequence))
    segments.append(("CDR2", cdr2.start, cdr2.sequence))
    segments.append(("CDR3", cdr3.start, cdr3.sequence))
    segments.append(("FR3", 1, full))   # framework concat — region label arbitrary

    chain_letter = "H" if numbering.chain_type == "H" else "L"

    # Generic motif passes
    for region_name, region_start, seq in segments:
        for motif_kind, pattern in _MOTIFS.items():
            for m in pattern.finditer(seq):
                # absolute position is 1-based within the *chain*
                pos = region_start + m.start()
                final_severity = _max_severity(
                    _MOTIF_BASE_SEVERITY[motif_kind],
                    _REGION_BASE_SEVERITY.get(region_name, "low"),
                )
                # Skip Met/Trp in framework (too noisy unless it's actually buried);
                # always report in CDRs.
                if motif_kind == "oxidation" and region_name not in {"CDR1", "CDR2", "CDR3"}:
                    continue
                out.append(Liability(
                    kind=motif_kind,
                    chain=chain_letter,
                    region=region_name,
                    position=pos,
                    motif=m.group(),
                    severity=final_severity,
                ))

    # Free-cysteine detection: Cys count must be even (paired). Report each
    # unpaired Cys with severity = high.
    for region_name, region_start, seq in segments:
        cys_positions = [region_start + i for i, aa in enumerate(seq) if aa == "C"]
        if len(cys_positions) % 2 == 1:
            # report only the *last* (likely unpaired) Cys to avoid dup spam
            pos = cys_positions[-1]
            out.append(Liability(
                kind="free_cysteine",
                chain=chain_letter,
                region=region_name,
                position=pos,
                motif="C",
                severity=_max_severity(
                    _MOTIF_BASE_SEVERITY["free_cysteine"],
                    _REGION_BASE_SEVERITY.get(region_name, "low"),
                ),
            ))

    return out


def run(
    vh_numbering: dict[str, Any],
    vl_numbering: dict[str, Any],
) -> dict[str, Any]:
    """Return the full liability report for both chains.

    Args:
        vh_numbering: a ChainNumbering dict (from `number_antibody`).
        vl_numbering: a ChainNumbering dict (from `number_antibody`).

    Returns:
        {
            "liabilities": [Liability dict, ...],
            "high_severity_count": int,
            "summary_by_kind": {kind: count},
        }
    """
    vh = ChainNumbering.model_validate(vh_numbering)
    vl = ChainNumbering.model_validate(vl_numbering)

    hits = _scan_chain(vh) + _scan_chain(vl)
    serialised = [h.model_dump() for h in hits]

    high = sum(1 for h in hits if h.severity == "high")
    summary: dict[str, int] = {}
    for h in hits:
        summary[h.kind] = summary.get(h.kind, 0) + 1

    return {
        "liabilities": serialised,
        "high_severity_count": high,
        "summary_by_kind": summary,
    }
