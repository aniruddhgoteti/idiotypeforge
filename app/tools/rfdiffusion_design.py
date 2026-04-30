"""RFdiffusion + ProteinMPNN binder design tool.

Designs *de novo* protein binders against the patient's idiotype CDR3
surface. Requires GPU for real runs; ships with a deterministic mock for
local methodology validation.

Toggle: env `IDIOTYPEFORGE_USE_MOCKS` (default = "1" → mock mode).

Real implementation (Day 13 on GCP A100 spot):
  1. RFdiffusion: generate scaffolds against the idiotype hotspots
     `--inference.input_pdb=...` `--ppi.hotspot_res=...`
  2. ProteinMPNN: design sequences for each scaffold, keep top by log-prob
  3. (Optional) AlphaFold-Multimer rescore → see `rescore_complex.py`
"""
from __future__ import annotations

from typing import Any

from ._mocks import mock_rfdiffusion_design, use_mocks


SCHEMA = {
    "name": "design_binder",
    "description": (
        "De-novo design protein binders against the patient's idiotype using "
        "RFdiffusion (scaffold generation) + ProteinMPNN (sequence design). "
        "Returns ranked candidates by ProteinMPNN log-probability."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_pdb": {
                "type": "string",
                "description": "PDB text of the patient's Fv (target = idiotype CDR3 surface).",
            },
            "hotspot_residues": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "1-based residue indices on the target to bind (typically CDR3-H).",
            },
            "n_designs": {"type": "integer", "default": 10},
        },
        "required": ["target_pdb", "hotspot_residues"],
    },
}


def run(
    target_pdb: str,
    hotspot_residues: list[int],
    n_designs: int = 10,
) -> dict[str, Any]:
    """Return ranked binder candidates.

    Mock mode (default for Days 1–12): returns `n_designs` candidates with
    realistic-shaped scaffolds + log-probs derived deterministically from the
    target PDB string + hotspot list. Sufficient to exercise the agent loop,
    UI, dossier prompt, and verification harness end-to-end.
    """
    if use_mocks():
        candidates = mock_rfdiffusion_design(
            target_pdb=target_pdb,
            hotspot_residues=hotspot_residues,
            n_designs=n_designs,
        )
        return {
            "candidates": candidates,
            "mock": True,
            "note": "MOCK output — set IDIOTYPEFORGE_USE_MOCKS=0 for real RFdiffusion.",
        }

    return _run_real_rfdiffusion(
        target_pdb=target_pdb,
        hotspot_residues=hotspot_residues,
        n_designs=n_designs,
    )


def _run_real_rfdiffusion(
    target_pdb: str,
    hotspot_residues: list[int],
    n_designs: int,
) -> dict[str, Any]:
    """Real GPU implementation. Day-13 on GCP A100 spot.

    Stub. Calls into RFdiffusion CLI:
        scripts/run_inference.py
            inference.input_pdb=$TARGET
            ppi.hotspot_res=[A30,A31,...]
            inference.num_designs=$n_designs
            denoiser.noise_scale_ca=0
            denoiser.noise_scale_frame=0
    Then runs ProteinMPNN on each scaffold and aggregates candidates.
    """
    raise NotImplementedError("Real RFdiffusion runs only on Day 13+ (GCP A100).")
