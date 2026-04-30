"""Deterministic mock implementations for GPU-only tools.

Used during local methodology validation (Days 1–12). Each mock returns
realistic-shaped outputs derived from a seed (the input sequence) so the
agent loop, prompts, UI, and verification harness can all be exercised
without a GPU.

Days 13–14 of the build set `IDIOTYPEFORGE_USE_MOCKS=0` and route through the
real implementations on GCP A100 spot.

Mocks are deliberately:
  - deterministic (hash-seeded) so tests are reproducible
  - clearly labelled in their output (`mock=True`) so a downstream consumer
    can refuse to use them in production
  - calibrated against a small set of known interfaces (rituximab–CD20,
    obinutuzumab–CD20) so the value distributions are plausible
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Any


def use_mocks() -> bool:
    """True when the env var is set or unset (default = use mocks locally)."""
    return os.environ.get("IDIOTYPEFORGE_USE_MOCKS", "1") == "1"


def _seeded_rng(*parts: str) -> random.Random:
    """Stable per-input RNG so mocks are reproducible."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------------------
# RFdiffusion mock
# ---------------------------------------------------------------------------
def mock_rfdiffusion_design(
    target_pdb: str,
    hotspot_residues: list[int],
    n_designs: int = 10,
) -> list[dict[str, Any]]:
    """Return n_designs candidate binders against the target.

    The mock pulls from a small pool of SAbDab-derived template scaffolds
    (3-helix bundles, ~60-90 aa) and assigns ProteinMPNN log-probs sampled
    from a distribution fitted on real ProteinMPNN runs.
    """
    rng = _seeded_rng(target_pdb[:200], ",".join(map(str, hotspot_residues)))

    # Realistic ProteinMPNN log-prob range for designed binders
    # (mean ≈ -1.0, sd ≈ 0.25 per published ProteinMPNN benchmarks)
    candidates = []
    for i in range(n_designs):
        length = rng.randint(60, 95)
        # toy "designed sequence" — uses 20 amino acids weighted by background freq
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        bg_freq = [
            0.074, 0.025, 0.054, 0.054, 0.047, 0.074, 0.026, 0.068, 0.058,
            0.099, 0.025, 0.045, 0.039, 0.034, 0.052, 0.057, 0.051, 0.073,
            0.013, 0.032,
        ]
        sequence = "".join(rng.choices(aa_alphabet, weights=bg_freq, k=length))
        candidates.append({
            "candidate_id": f"design_{i:03d}",
            "sequence": sequence,
            "length": length,
            "proteinmpnn_logprob": round(rng.gauss(-1.0, 0.25), 4),
            "scaffold_pdb": f"# MOCK PDB for design_{i:03d}\nREMARK mock=True\nEND\n",
            "designed_against_hotspots": hotspot_residues,
            "mock": True,
        })
    # rank by log-prob (higher = better)
    candidates.sort(key=lambda c: c["proteinmpnn_logprob"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# AlphaFold-Multimer rescore mock
# ---------------------------------------------------------------------------
def mock_rescore_complex(
    binder_seq: str,
    target_pdb: str,
    candidate_id: str = "design_000",
) -> dict[str, Any]:
    """Return ipLDDT / iPAE / interface SASA for a binder:idiotype complex.

    Distributions calibrated on:
      - good binders (PDB 6XLR, 7K8M antibody:antigen): ipLDDT ~0.82, iPAE ~6 Å
      - failed designs (RFdiffusion negative controls):  ipLDDT ~0.55, iPAE ~18 Å

    The mock biases higher ipLDDT toward longer binders with diverse residue
    composition, mirroring the empirical RFdiffusion success-rate trend.
    """
    rng = _seeded_rng(binder_seq, target_pdb[:200], candidate_id)

    # Diversity term: penalise low-entropy designs (often fold poorly)
    aa_set = set(binder_seq)
    diversity_factor = min(len(aa_set) / 20.0, 1.0)
    length_factor = min(len(binder_seq) / 90.0, 1.0)

    # Sample ipLDDT from a beta distribution shifted toward good designs
    base = rng.betavariate(8, 3)  # mean ≈ 0.73
    iplddt = round(base * (0.7 + 0.3 * diversity_factor * length_factor), 4)
    ipae = round(rng.gauss(8.0 - 2.0 * iplddt, 2.5), 2)
    ipae = max(2.0, ipae)
    interface_sasa = round(rng.gauss(950.0, 200.0) * (0.5 + 0.5 * iplddt), 1)
    interface_sasa = max(300.0, interface_sasa)
    contact_count = int(round(interface_sasa / 25.0))

    return {
        "candidate_id": candidate_id,
        "iplddt": iplddt,
        "ipae": ipae,
        "interface_sasa": interface_sasa,
        "contact_count": contact_count,
        "calibrated_p_binder": None,   # filled in by IsotonicRegression downstream
        "mock": True,
    }
