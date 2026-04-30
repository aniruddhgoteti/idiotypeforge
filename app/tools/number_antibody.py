"""ANARCI-based antibody numbering tool.

Wraps the ANARCI Python API to assign Kabat/IMGT/Chothia numbering to a
paired Fv (VH + VL), extract CDR1/2/3 spans, and infer V/J gene + isotype.

CPU-only. Requires `anarci` (pip) + HMMER (system binary).

Lazy import: the heavy `anarci` import happens inside `_run_anarci()` so the
module loads in environments where ANARCI isn't installed (the rest of the
toolchain still works in stub mode).

Verification target (per plan §4):
    100% match to Kabat reference for rituximab (PDB 1N8Z) + trastuzumab.

Reference: Dunbar & Deane 2016 *Bioinformatics* "ANARCI: antigen receptor
numbering and receptor classification".
"""
from __future__ import annotations

from typing import Any

from ._types import CDRSpan, ChainNumbering


SCHEMA = {
    "name": "number_antibody",
    "description": (
        "Assign Kabat / IMGT / Chothia numbering to an antibody Fv. Returns "
        "CDR1/2/3 spans, framework sequences, and V/J gene + isotype "
        "assignments for both chains."
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


# ---------------------------------------------------------------------------
# CDR boundary definitions
# ---------------------------------------------------------------------------
# Kabat numbering uses these inclusive position ranges for CDRs:
#   VH:  CDR1 = 31–35,  CDR2 = 50–65,  CDR3 = 95–102
#   VL:  CDR1 = 24–34,  CDR2 = 50–56,  CDR3 = 89–97
# IMGT differs:
#   VH:  CDR1 = 27–38,  CDR2 = 56–65,  CDR3 = 105–117
#   VL:  CDR1 = 27–38,  CDR2 = 56–65,  CDR3 = 105–117
# Chothia is similar to Kabat for FR/CDR boundaries except CDR1.
# ---------------------------------------------------------------------------
_CDR_RANGES: dict[tuple[str, str], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
    ("kabat",   "H"): ((31, 35), (50, 65), (95, 102)),
    ("kabat",   "K"): ((24, 34), (50, 56), (89, 97)),
    ("kabat",   "L"): ((24, 34), (50, 56), (89, 97)),
    ("imgt",    "H"): ((27, 38), (56, 65), (105, 117)),
    ("imgt",    "K"): ((27, 38), (56, 65), (105, 117)),
    ("imgt",    "L"): ((27, 38), (56, 65), (105, 117)),
    ("chothia", "H"): ((26, 32), (52, 56), (95, 102)),
    ("chothia", "K"): ((24, 34), (50, 56), (89, 97)),
    ("chothia", "L"): ((24, 34), (50, 56), (89, 97)),
}


def run(
    vh_sequence: str,
    vl_sequence: str,
    scheme: str = "kabat",
) -> dict[str, Any]:
    """Number both chains of the Fv and return JSON-serialisable CDR spans."""
    if scheme not in {"kabat", "imgt", "chothia"}:
        raise ValueError(f"Unsupported scheme: {scheme}")

    vh_seq = vh_sequence.strip().upper()
    vl_seq = vl_sequence.strip().upper()

    if not vh_seq or not vl_seq:
        raise ValueError("Both vh_sequence and vl_sequence are required.")

    vh_numbering = _number_chain(vh_seq, scheme=scheme, is_heavy=True)
    vl_numbering = _number_chain(vl_seq, scheme=scheme, is_heavy=False)

    return {
        "vh": vh_numbering.model_dump(),
        "vl": vl_numbering.model_dump(),
    }


# ---------------------------------------------------------------------------
# ANARCI integration (lazy import)
# ---------------------------------------------------------------------------
def _number_chain(seq: str, scheme: str, is_heavy: bool) -> ChainNumbering:
    """Run ANARCI on a single chain and parse the output into a ChainNumbering."""
    try:
        from anarci import run_anarci  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "ANARCI is not installed. Install with: `uv pip install anarci`. "
            "ANARCI also requires HMMER (system binary) — `brew install hmmer` on "
            "macOS or `apt install hmmer` on Debian/Ubuntu."
        ) from e

    anarci_input = [(f"chain_{int(is_heavy)}", seq)]
    # ANARCI returns: numbered, alignment_details, hit_tables
    numbered, details, _ = run_anarci(anarci_input, scheme=scheme.upper(), assign_germline=True)

    if not numbered or numbered[0] is None:
        raise ValueError(
            f"ANARCI failed to recognise the supplied {'heavy' if is_heavy else 'light'} "
            "chain. Check that you supplied a variable-region sequence (~110–130 aa)."
        )

    # numbered[0] is a list of (numbered_residues, start_idx, end_idx) tuples;
    # we take the first (top-scoring) domain.
    chain_records = numbered[0]
    residues, _start_idx, _end_idx = chain_records[0]
    detail = details[0][0] if details and details[0] else {}

    chain_type = _resolve_chain_type(detail, is_heavy)
    return _residues_to_chain_numbering(
        residues=residues,
        scheme=scheme,
        chain_type=chain_type,
        v_gene=detail.get("germlines", {}).get("v_gene", [(None,)])[0][1] if detail.get("germlines") else None,
        j_gene=detail.get("germlines", {}).get("j_gene", [(None,)])[0][1] if detail.get("germlines") else None,
    )


def _resolve_chain_type(detail: dict[str, Any], is_heavy: bool) -> str:
    """Determine 'H' / 'K' / 'L' from ANARCI alignment details."""
    chain_type = detail.get("chain_type")
    if chain_type in {"H", "K", "L"}:
        return chain_type
    return "H" if is_heavy else "K"


def _residues_to_chain_numbering(
    residues: list[tuple[tuple[int, str], str]],
    scheme: str,
    chain_type: str,
    v_gene: str | None,
    j_gene: str | None,
) -> ChainNumbering:
    """Convert ANARCI's residue list into typed CDR spans + framework string.

    `residues` looks like:  [((1, ' '), 'E'), ((2, ' '), 'V'), ..., ((114, ' '), 'S')]
    Gaps appear as `'-'`.
    """
    cdr_ranges = _CDR_RANGES.get((scheme, chain_type))
    if cdr_ranges is None:
        raise ValueError(f"No CDR ranges defined for scheme={scheme} chain={chain_type}")
    (cdr1_lo, cdr1_hi), (cdr2_lo, cdr2_hi), (cdr3_lo, cdr3_hi) = cdr_ranges

    cdr1_aa: list[str] = []
    cdr2_aa: list[str] = []
    cdr3_aa: list[str] = []
    fr_aa: list[str] = []

    cdr1_first_pos: int | None = None
    cdr2_first_pos: int | None = None
    cdr3_first_pos: int | None = None

    # ANARCI emits residues in order; absolute 1-based "linear" position is
    # the index into the non-gap residues list.
    linear_pos = 0
    for (kabat_pos, _ins), aa in residues:
        if aa == "-":
            continue
        linear_pos += 1
        if cdr1_lo <= kabat_pos <= cdr1_hi:
            if cdr1_first_pos is None:
                cdr1_first_pos = linear_pos
            cdr1_aa.append(aa)
        elif cdr2_lo <= kabat_pos <= cdr2_hi:
            if cdr2_first_pos is None:
                cdr2_first_pos = linear_pos
            cdr2_aa.append(aa)
        elif cdr3_lo <= kabat_pos <= cdr3_hi:
            if cdr3_first_pos is None:
                cdr3_first_pos = linear_pos
            cdr3_aa.append(aa)
        else:
            fr_aa.append(aa)

    if not cdr3_aa or cdr3_first_pos is None:
        raise ValueError(
            f"Failed to find CDR3 in {chain_type}-chain numbering. ANARCI output may "
            "be malformed; check input is a real variable region."
        )

    return ChainNumbering(
        chain_type=chain_type,
        scheme=scheme,
        v_gene=v_gene,
        j_gene=j_gene,
        isotype=None,        # isotype lives in the constant region; not numbered here
        cdr1=CDRSpan(
            start=cdr1_first_pos or 0,
            end=(cdr1_first_pos or 0) + len(cdr1_aa) - 1,
            sequence="".join(cdr1_aa),
        ),
        cdr2=CDRSpan(
            start=cdr2_first_pos or 0,
            end=(cdr2_first_pos or 0) + len(cdr2_aa) - 1,
            sequence="".join(cdr2_aa),
        ),
        cdr3=CDRSpan(
            start=cdr3_first_pos,
            end=cdr3_first_pos + len(cdr3_aa) - 1,
            sequence="".join(cdr3_aa),
        ),
        framework_sequence="".join(fr_aa),
    )
