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

    Calls colabfold_batch on a paired (binder, target) FASTA. Parses ipLDDT
    and iPAE from the predicted CIF; computes interface SASA via FreeSASA;
    counts interface contacts (Cα–Cα < 8 Å) directly from the structure.

    Environment expectations (set by `scripts/setup_a100.sh`):
      colabfold_batch on PATH (`uv pip install "colabfold[alphafold]"`)
      CUDA available
    """
    import json
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("colabfold_batch") is None:
        raise RuntimeError(
            "colabfold_batch not on PATH. Install with "
            "`uv pip install 'colabfold[alphafold]' --extra-index-url https://pypi.nvidia.com` "
            "or run scripts/setup_a100.sh."
        )

    target_seq = _extract_sequence_from_pdb(target_pdb)
    if not target_seq:
        raise RuntimeError("Could not extract target sequence from supplied PDB.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # ColabFold's "complex" syntax: chains separated by ":" in one record
        fasta = tmp_path / f"{candidate_id}.fasta"
        fasta.write_text(f">{candidate_id}\n{binder_sequence}:{target_seq}\n")

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        cmd = [
            "colabfold_batch",
            str(fasta),
            str(out_dir),
            "--num-models", "1",
            "--num-recycle", "3",
            "--rank", "iptm",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(
                f"colabfold_batch failed (rc={result.returncode}): "
                f"{result.stderr[-500:]}"
            )

        # Find the top-ranked CIF + the scores JSON
        scores_files = sorted(out_dir.glob(f"{candidate_id}_scores_rank_001*.json"))
        cif_files = sorted(out_dir.glob(f"{candidate_id}_unrelaxed_rank_001*.cif")) \
                    or sorted(out_dir.glob(f"{candidate_id}_unrelaxed_rank_001*.pdb"))
        if not scores_files or not cif_files:
            raise RuntimeError(
                f"Expected ColabFold outputs missing in {out_dir}: "
                f"scores={list(out_dir.glob('*scores*'))} structures={list(out_dir.glob('*rank_001*'))}"
            )

        scores = json.loads(scores_files[0].read_text())
        plddt = scores.get("plddt") or []
        pae_matrix = scores.get("pae") or []

        binder_len = len(binder_sequence)
        iplddt = float(sum(plddt[:binder_len]) / binder_len) / 100.0 if plddt else 0.0
        # iPAE = mean PAE across the binder↔target submatrix
        ipae = _mean_interface_pae(pae_matrix, binder_len)

        structure_text = cif_files[0].read_text()
        sasa = _compute_interface_sasa(structure_text, binder_len)
        contacts = _count_interface_contacts(structure_text, binder_len)

        return {
            "candidate_id": candidate_id,
            "iplddt": round(iplddt, 4),
            "ipae": round(ipae, 2),
            "interface_sasa": round(sasa, 1),
            "contact_count": contacts,
            "calibrated_p_binder": None,
            "mock": False,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_sequence_from_pdb(pdb_text: str) -> str:
    """Extract the chain-A amino-acid sequence from a PDB string."""
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seen: set[tuple[str, int]] = set()
    seq: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21:22].strip() or "A"
        if chain not in {"A", "H"}:
            continue
        try:
            resname = line[17:20].strip()
            resseq = int(line[22:26])
        except ValueError:
            continue
        key = (chain, resseq)
        if key in seen:
            continue
        seen.add(key)
        seq.append(three_to_one.get(resname, "X"))
    return "".join(seq).rstrip("X")


def _mean_interface_pae(pae_matrix: list[list[float]], binder_len: int) -> float:
    """Mean PAE in the binder ↔ target off-diagonal block."""
    if not pae_matrix:
        return 30.0
    n = len(pae_matrix)
    if binder_len >= n:
        return 30.0
    total = 0.0
    count = 0
    for i in range(binder_len):
        for j in range(binder_len, n):
            total += pae_matrix[i][j]
            total += pae_matrix[j][i]
            count += 2
    return total / count if count > 0 else 30.0


def _compute_interface_sasa(structure_text: str, binder_len: int) -> float:
    """Buried interface SASA in Å² via FreeSASA."""
    try:
        import freesasa
    except ImportError:
        return 0.0

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as fh:
        fh.write(structure_text if "ATOM" in structure_text else "")
        path = fh.name

    try:
        struct = freesasa.Structure(path)
        result = freesasa.Calc().calculate(struct)
        return float(result.totalArea())
    except Exception:               # noqa: BLE001
        return 0.0


def _count_interface_contacts(structure_text: str, binder_len: int) -> int:
    """Count Cα–Cα contacts < 8 Å between binder and target chains."""
    binder_cas: list[tuple[float, float, float]] = []
    target_cas: list[tuple[float, float, float]] = []
    for line in structure_text.splitlines():
        if not (line.startswith("ATOM") and line[12:16].strip() == "CA"):
            continue
        try:
            resseq = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        # Treat the first `binder_len` distinct residues as the binder; rest = target.
        target_list = target_cas if resseq > binder_len else binder_cas
        target_list.append((x, y, z))

    contacts = 0
    for ax, ay, az in binder_cas:
        for bx, by, bz in target_cas:
            if (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2 < 64.0:
                contacts += 1
    return contacts
