"""ANARCI-based antibody numbering tool.

Wraps the ANARCI CLI / Python API to assign Kabat (default) or IMGT numbering
to a paired Fv (VH + VL), extract CDR1/2/3 spans, and infer V/J gene + isotype.

CPU-runnable. Pure Python + HMMER under the hood.

Verification target (per plan §4):
    100% match to Kabat reference for rituximab (PDB 1N8Z) + trastuzumab.
"""
from __future__ import annotations

from typing import Any

from ._types import CDRSpan, ChainNumbering


SCHEMA = {
    "name": "number_antibody",
    "description": (
        "Assign Kabat / IMGT numbering to an antibody Fv. Returns CDR1/2/3 spans, "
        "framework sequences, and V/J gene + isotype assignments for both chains."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "vh_sequence": {
                "type": "string",
                "description": "Heavy-chain variable region amino-acid sequence.",
            },
            "vl_sequence": {
                "type": "string",
                "description": "Light-chain variable region amino-acid sequence.",
            },
            "scheme": {
                "type": "string",
                "enum": ["kabat", "imgt", "chothia"],
                "default": "kabat",
            },
        },
        "required": ["vh_sequence", "vl_sequence"],
    },
}


def run(
    vh_sequence: str,
    vl_sequence: str,
    scheme: str = "kabat",
) -> dict[str, Any]:
    """Number the Fv and return JSON-serialisable CDR spans + framework.

    Day-2 implementation:
        - import anarci.run_anarci
        - call once per chain with `scheme=scheme.upper()`
        - parse numbered output into CDRSpan objects (positions per IMGT/Kabat)
        - infer V/J gene + isotype from the germline assignment column
    """
    raise NotImplementedError(
        "Stub: implement on Day 2. Reference: anarci.run_anarci API. "
        "Test against rituximab and trastuzumab Kabat numbering."
    )


def _parse_anarci_chain(numbered: list[tuple[Any, str]], scheme: str) -> ChainNumbering:
    """Helper: turn ANARCI numbered output into a ChainNumbering.

    Stub.
    """
    raise NotImplementedError
