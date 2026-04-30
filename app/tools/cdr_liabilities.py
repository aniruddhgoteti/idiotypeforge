"""CDR liability scanner.

Scans CDR loops + framework for known developability liabilities:
  - N-glycosylation motif (N-X-S/T, X != P)
  - Asn deamidation hotspots (NG, NS)
  - Asp isomerization (DG, DS, DT, DD, DH)
  - Met / Trp oxidation
  - Free cysteines (unpaired Cys)
  - Asp-Pro fragmentation (DP)

Pure-python regex over the numbered chain. CPU-runnable.

Verification target (per plan §4):
    Reproduce the published liability profile of trastuzumab (TRA-12) and
    omalizumab (free Cys at H85) on the corresponding test cases.
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


# Severity weights — a CDR3 hotspot is more severe than a buried framework one.
_REGION_SEVERITY = {
    "FR1": "low", "FR2": "low", "FR3": "low", "FR4": "low",
    "CDR1": "medium", "CDR2": "medium", "CDR3": "high",
}


def run(
    vh_numbering: dict[str, Any],
    vl_numbering: dict[str, Any],
) -> dict[str, Any]:
    """Return the full liability report for both chains.

    Day-2 implementation:
        - parse vh_numbering / vl_numbering as ChainNumbering
        - for each region (FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4) walk sequence
          and apply each motif regex
        - severity from _REGION_SEVERITY
        - return list of Liability dicts
    """
    raise NotImplementedError("Stub: implement on Day 2.")


# ---------------------------------------------------------------------------
# Motif regexes (compile once)
# ---------------------------------------------------------------------------
_MOTIFS = {
    "n_glycosylation": re.compile(r"N[^P][ST]"),
    "deamidation":     re.compile(r"N[GS]"),
    "isomerization":   re.compile(r"D[GSTDH]"),
    "oxidation":       re.compile(r"[MW]"),
    "free_cysteine":   re.compile(r"C"),
    "fragmentation":   re.compile(r"DP"),
}
