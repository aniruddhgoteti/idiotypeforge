"""IgFold-based Fv structure prediction tool.

Falls back to ESMFold if IgFold is unavailable. CPU-runnable but slow:
~2–5 min/seq on Apple-silicon CPU; ~15-30 s on a single L4/A100.

Verification target (per plan §4):
    mean Fv pLDDT ≥ 0.85; on 5 SAbDab antibodies w/ crystal structures:
    framework RMSD ≤ 1.5 Å, CDR3 RMSD ≤ 3.0 Å.
"""
from __future__ import annotations

from typing import Any

from ._types import FvStructure


SCHEMA = {
    "name": "predict_fv_structure",
    "description": (
        "Predict the 3D structure of an antibody Fv (VH + VL) using IgFold. "
        "Returns a PDB string, per-residue pLDDT, mean pLDDT, CDR3-mean pLDDT, "
        "and a base64-encoded PNG render."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "vh_sequence": {"type": "string"},
            "vl_sequence": {"type": "string"},
            "render": {
                "type": "boolean",
                "default": True,
                "description": "If true, attach a base64 PNG of the structure.",
            },
        },
        "required": ["vh_sequence", "vl_sequence"],
    },
}


def run(
    vh_sequence: str,
    vl_sequence: str,
    render: bool = True,
) -> dict[str, Any]:
    """Predict the Fv structure and return a JSON-serialisable FvStructure.

    Day-2 implementation:
        - lazily import igfold (so base install stays light)
        - call IgFoldRunner().fold(...)
        - read predicted pLDDT from the per-residue B-factor column
        - if `render`, call render_structure tool to attach a PNG
        - on CPU, set num_threads = os.cpu_count() and emit a slow-mode warning
    """
    raise NotImplementedError(
        "Stub: implement on Day 2. Use IgFoldRunner with `do_refine=False` on "
        "CPU to keep runtime reasonable. Reference: PDB 1N8Z (rituximab) for "
        "verification."
    )
