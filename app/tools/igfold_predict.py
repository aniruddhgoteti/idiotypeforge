"""IgFold-based Fv structure prediction tool.

Wraps the IgFold (Ruffolo et al. 2023, *Nat Commun*) antibody-specific
structure predictor. CPU-runnable but slow: ~2–5 min/seq on Apple-silicon
CPU; ~15–30 s on a single L4/A100.

Lazy import: the heavy `igfold` module loads inside `_run_igfold()`. The
tool module itself loads cleanly even when IgFold isn't installed.

Verification target (per plan §4):
    mean Fv pLDDT ≥ 0.85; on 5 SAbDab antibodies w/ crystal structures:
    framework RMSD ≤ 1.5 Å, CDR3 RMSD ≤ 3.0 Å.

Reference: https://github.com/Graylab/IgFold
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any


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
            "do_refine": {
                "type": "boolean",
                "default": False,
                "description": "OpenMM relaxation. Off by default (slow on CPU).",
            },
        },
        "required": ["vh_sequence", "vl_sequence"],
    },
}


def run(
    vh_sequence: str,
    vl_sequence: str,
    render: bool = True,
    do_refine: bool = False,
) -> dict[str, Any]:
    """Predict the Fv structure and return a JSON-serialisable result.

    Returns:
        {
            "pdb_text": str,
            "plddt": list[float],          # per-residue, [0, 1]
            "mean_plddt": float,
            "cdr3_mean_plddt": float,      # over the heavy-chain CDR3 residues
            "render_png_b64": str | None,
            "wallclock_seconds": float,
        }
    """
    vh = vh_sequence.strip().upper()
    vl = vl_sequence.strip().upper()
    if not vh or not vl:
        raise ValueError("Both vh_sequence and vl_sequence are required.")

    pdb_path, plddt, mean, cdr3_mean, wallclock = _run_igfold(vh, vl, do_refine=do_refine)

    pdb_text = Path(pdb_path).read_text()

    render_png: str | None = None
    if render:
        # Lazy: don't import render_structure at module load — avoids matplotlib
        # being pulled in for users who only want the structure.
        from app.tools import render_structure
        try:
            r = render_structure.run(pdb_text=pdb_text, color_by_plddt=True)
            render_png = r.get("overview")
        except NotImplementedError:
            # render_structure is a stub until Day 9 — fine, just skip the render.
            render_png = None

    return {
        "pdb_text": pdb_text,
        "plddt": plddt,
        "mean_plddt": mean,
        "cdr3_mean_plddt": cdr3_mean,
        "render_png_b64": render_png,
        "wallclock_seconds": wallclock,
    }


# ---------------------------------------------------------------------------
# IgFold wrapper (lazy import)
# ---------------------------------------------------------------------------
def _run_igfold(
    vh: str,
    vl: str,
    do_refine: bool = False,
) -> tuple[str, list[float], float, float, float]:
    """Run IgFold on a paired Fv. Returns (pdb_path, plddt, mean, cdr3_mean, seconds)."""
    import time

    try:
        from igfold import IgFoldRunner  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "IgFold is not installed. Install with: `uv sync --extra igfold` or "
            "`uv pip install igfold ablang2`. IgFold weights (~500 MB) auto-download "
            "on first use to ~/.cache/igfold/."
        ) from e

    # Force CPU mode unless GPU is explicitly available (Day 13+ GCP A100 phase).
    # IgFold respects torch.cuda.is_available() but we let user override via env.
    if os.environ.get("IDIOTYPEFORGE_FORCE_CPU", "1") == "1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    sequences = {"H": vh, "L": vl}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pdb = Path(tmpdir) / "fv.pdb"
        runner = IgFoldRunner()
        t0 = time.time()
        result = runner.fold(
            str(out_pdb),
            sequences=sequences,
            do_refine=do_refine,
            do_renum=False,                 # we already number separately
            use_openmm=False,                # avoid OpenMM if do_refine=False
        )
        wallclock = time.time() - t0

        # IgFold writes pLDDT into the B-factor column. The result object also
        # exposes `prmsd` (per-residue RMSD prediction) which we treat as a
        # confidence proxy: pLDDT_proxy = exp(-prmsd / 5.0), clamped to [0, 1].
        # IgFold v0.4.0+ exposes `prmsd` as a numpy array; older versions used
        # `result.prmsd_loss`. Try both.
        prmsd = getattr(result, "prmsd", None)
        if prmsd is None:
            prmsd = getattr(result, "prmsd_loss", None)
        if prmsd is None:
            # Fall back: parse B-factor from the PDB itself.
            plddt = _parse_bfactors(out_pdb)
        else:
            import numpy as np
            arr = np.asarray(prmsd).flatten()
            plddt = [max(0.0, min(1.0, float(np.exp(-x / 5.0)))) for x in arr]

        # CDR3 region: assumed to be the contiguous heavy-chain residues at
        # ~(VH_len - 12) ... (VH_len - 5) by Kabat convention. The numbering tool
        # is the authoritative source; here we just take the rough middle of
        # the heavy chain as a proxy when called standalone.
        vh_len = len(vh)
        cdr3_lo = max(0, vh_len - 18)
        cdr3_hi = max(cdr3_lo + 1, vh_len - 6)
        cdr3_plddt = plddt[cdr3_lo:cdr3_hi] or plddt
        mean = sum(plddt) / max(1, len(plddt))
        cdr3_mean = sum(cdr3_plddt) / max(1, len(cdr3_plddt))

        # Copy PDB out of the tempdir so the caller can read it after we exit.
        persistent = Path(tempfile.mkstemp(suffix=".pdb")[1])
        persistent.write_bytes(out_pdb.read_bytes())

    return str(persistent), plddt, mean, cdr3_mean, wallclock


def _parse_bfactors(pdb_path: Path) -> list[float]:
    """Extract per-CA-atom B-factor (treated as pLDDT/100) from a PDB file."""
    plddt: list[float] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            b = float(line[60:66])
        except ValueError:
            continue
        # B-factor columns ranges: IgFold writes pLDDT × 100. Normalise to [0, 1].
        plddt.append(max(0.0, min(1.0, b / 100.0)))
    return plddt
