"""CAR-T cassette assembler.

Wraps the chosen scFv (VH-(G4S)3-VL) inside a standard 2nd-generation CAR
backbone. Two formats supported:

  CD28z   — CD8α leader · scFv · CD8α hinge · CD28 TM · CD28 cyto · CD3ζ
  4-1BBz  — CD8α leader · scFv · CD8α hinge · CD28 TM · 4-1BB cyto · CD3ζ
            (this is the tisagenlecleucel / Maude 2018 NEJM construct)

CPU-only, pure-string assembly. Returns full amino-acid sequence + a
component dict and a `plasmid_map.gb`-friendly description.
"""
from __future__ import annotations

from typing import Any

from ._types import CARConstruct


# ---------------------------------------------------------------------------
# Reference sequences (canonical, public, redistributable)
# Source: GenBank / Maude 2018 NEJM tisagenlecleucel construct
# ---------------------------------------------------------------------------
CD8A_LEADER = "MALPVTALLLPLALLLHAARP"
G4S_LINKER_3 = "GGGGSGGGGSGGGGS"
CD8A_HINGE = (
    "TTTPAPRPPTPAPTIASQPLSLRPEACRPAAGGAVHTRGLDFACD"
)
CD28_TM = "FWVLVVVGGVLACYSLLVTVAFIIFWV"
CD28_CYTO = "RSKRSRLLHSDYMNMTPRRPGPTRKHYQPYAPPRDFAAYRS"
FOURONEBB_CYTO = "KRGRKKLLYIFKQPFMRPVQTTQEEDGCSCRFPEEEEGGCEL"
CD3Z = (
    "RVKFSRSADAPAYKQGQNQLYNELNLGRREEYDVLDKRRGRDPEMGGKPRRKNPQEGLYNELQKDKMAEAY"
    "SEIGMKGERRRGKGHDGLYQGLSTATKDTYDALHMQALPPR"
)


SCHEMA = {
    "name": "assemble_car_construct",
    "description": (
        "Assemble a 2nd-generation CAR-T cassette around the chosen scFv. "
        "Returns the full CAR amino-acid sequence and a component breakdown."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scfv_vh": {"type": "string"},
            "scfv_vl": {"type": "string"},
            "format": {
                "type": "string",
                "enum": ["CD28z", "4-1BBz"],
                "default": "4-1BBz",
            },
        },
        "required": ["scfv_vh", "scfv_vl"],
    },
}


def run(
    scfv_vh: str,
    scfv_vl: str,
    format: str = "4-1BBz",
) -> dict[str, Any]:
    """Assemble the cassette and return a CARConstruct dict."""
    if format not in {"CD28z", "4-1BBz"}:
        raise ValueError(f"Unsupported format: {format}")

    scfv = scfv_vh + G4S_LINKER_3 + scfv_vl
    costim = CD28_CYTO if format == "CD28z" else FOURONEBB_CYTO
    full = CD8A_LEADER + scfv + CD8A_HINGE + CD28_TM + costim + CD3Z

    return CARConstruct(
        format=format,
        full_aa_sequence=full,
        components={
            "leader": CD8A_LEADER,
            "scFv_VH": scfv_vh,
            "linker": G4S_LINKER_3,
            "scFv_VL": scfv_vl,
            "hinge": CD8A_HINGE,
            "TM": CD28_TM,
            "costim": costim,
            "CD3z": CD3Z,
        },
    ).model_dump()
