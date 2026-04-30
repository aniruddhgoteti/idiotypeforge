"""Off-target safety scan.

Two databases:
  1. **OAS healthy paired** (~5–10 M sequences, downloaded by
     `scripts/download_oas.py`) — confirms the patient's idiotype is not
     similar to a normal B-cell clone in another individual.
  2. **UniProt human SwissProt** (~20 K sequences) — confirms the binder
     does not cross-react with any human protein.

Both searches use system binaries (MMseqs2 + BLAST). CPU-only.

Verification target (per plan §4):
    max identity to OAS healthy CDR3 < 70 %; FPR vs. random scrambled CDR3
    ≤ 0.01 %.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ._types import OffTargetHit, OffTargetReport


SCHEMA = {
    "name": "offtarget_search",
    "description": (
        "Search OAS healthy paired antibodies + UniProt human proteome for "
        "sequence matches to a candidate epitope or designed binder. Returns "
        "max identity, count of high-identity hits, and the top hits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_sequence": {
                "type": "string",
                "description": "Either a CDR3 peptide or a full designed binder sequence.",
            },
            "kind": {
                "type": "string",
                "enum": ["epitope", "binder"],
                "default": "binder",
            },
            "identity_threshold_pct": {"type": "number", "default": 70.0},
        },
        "required": ["query_sequence"],
    },
}


# Configurable via env: OAS_DB_DIR, UNIPROT_DB_DIR
_DEFAULT_OAS_DB = Path("data/oas/oas_healthy_paired.mmseqs")
_DEFAULT_UNIPROT_DB = Path("data/uniprot/uniprot_human_swissprot.blastdb")


def run(
    query_sequence: str,
    kind: str = "binder",
    identity_threshold_pct: float = 70.0,
) -> dict[str, Any]:
    """Run MMseqs2 + BLAST and aggregate the report.

    Day-3 implementation:
        - write `query_sequence` to a temp FASTA
        - call `mmseqs easy-search query.fa OAS_DB tmp/result.m8 tmp/`
        - call `blastp -query query.fa -db UNIPROT_DB -outfmt 6 -evalue 1e-3`
        - parse top hits, build OffTargetHit list
        - max_identity_pct = max identity across both DBs
    """
    raise NotImplementedError("Stub: implement on Day 3.")


def _parse_mmseqs_m8(path: Path) -> list[OffTargetHit]:
    """Helper. Stub."""
    raise NotImplementedError


def _parse_blast_outfmt6(path: Path) -> list[OffTargetHit]:
    """Helper. Stub."""
    raise NotImplementedError
