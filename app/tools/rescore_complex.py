"""AlphaFold-Multimer / Boltz-2 complex rescore.

Re-folds the (binder, idiotype) complex and computes interface metrics:
ipLDDT, iPAE, interface SASA, contact count. Used to rank RFdiffusion-
designed candidates.

Requires GPU for real runs; ships with a deterministic mock keyed on the
binder sequence so the same input always produces the same metrics.

Toggle: env `IDIOTYPEFORGE_USE_MOCKS` (default = "1" → mock mode).

Real implementation (Day 13 on GCP A100 spot):
  - colabfold-batch with --num-models 1 --num-recycle 3 on a single MSA
  - or Boltz-2 single-shot if A100 80 GB available
  - parse ipLDDT/iPAE from the predicted-PDB CIF
"""
from __future__ import annotations

from typing import Any

from ._mocks import mock_rescore_complex, use_mocks


SCHEMA = {
    "name": "rescore_complex",
    "description": (
        "Re-fold a (binder, idiotype) complex with AlphaFold-Multimer or Boltz-2 "
        "and return interface quality metrics: ipLDDT, iPAE, interface SASA, "
        "contact count. Used to rank designed binder candidates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "binder_sequence": {"type": "string"},
            "target_pdb": {"type": "string"},
            "candidate_id": {"type": "string"},
        },
        "required": ["binder_sequence", "target_pdb", "candidate_id"],
    },
}


def run(
    binder_sequence: str,
    target_pdb: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Return interface scores for a single binder:idiotype complex.

    Mock mode (default for Days 1–12): deterministic from inputs, calibrated
    on real AF-Multimer output distributions.
    """
    if use_mocks():
        return {
            **mock_rescore_complex(
                binder_seq=binder_sequence,
                target_pdb=target_pdb,
                candidate_id=candidate_id,
            ),
            "note": "MOCK output — set IDIOTYPEFORGE_USE_MOCKS=0 for real AF-Multimer.",
        }

    return _run_real_rescore(
        binder_sequence=binder_sequence,
        target_pdb=target_pdb,
        candidate_id=candidate_id,
    )


def _run_real_rescore(
    binder_sequence: str,
    target_pdb: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Real GPU implementation. Day-13 on GCP A100 spot.

    Stub. Calls colabfold_batch with both chains as a paired MSA, then parses
    PAE from the predicted CIF and computes interface SASA via FreeSASA.
    """
    raise NotImplementedError("Real AF-Multimer runs only on Day 13+ (GCP A100).")
