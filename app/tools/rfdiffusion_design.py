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

    Calls the RFdiffusion CLI at $RFDIFFUSION_DIR/scripts/run_inference.py to
    generate `n_designs` scaffold backbones against the target's hotspot
    residues, then runs ProteinMPNN on each scaffold to assign a sequence.

    Environment expectations (set by `scripts/setup_a100.sh`):
      RFDIFFUSION_DIR     — path to the cloned RFdiffusion repo
      PROTEINMPNN_DIR     — path to the cloned ProteinMPNN repo
      CUDA_VISIBLE_DEVICES — usually "0"

    The function shells out via subprocess; failures are translated into a
    RuntimeError with the last 500 chars of stderr so the orchestrator can
    surface a clear diagnostic.
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    rfd_dir = os.environ.get("RFDIFFUSION_DIR", "/opt/RFdiffusion")
    rfd_inference = Path(rfd_dir) / "scripts" / "run_inference.py"
    if not rfd_inference.exists():
        raise RuntimeError(
            f"RFdiffusion not found at {rfd_inference}. Set RFDIFFUSION_DIR or "
            "run scripts/setup_a100.sh on the VM."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target_pdb_path = tmp_path / "target.pdb"
        target_pdb_path.write_text(target_pdb)

        out_prefix = tmp_path / "design"
        hotspot_str = "[" + ",".join(f"A{r}" for r in hotspot_residues) + "]"

        cmd = [
            "python", str(rfd_inference),
            f"inference.input_pdb={target_pdb_path}",
            f"contigmap.contigs=[A1-130/0 60-90]",     # target chain A, designed binder length 60–90
            f"ppi.hotspot_res={hotspot_str}",
            f"inference.num_designs={n_designs}",
            f"inference.output_prefix={out_prefix}",
            "denoiser.noise_scale_ca=0",
            "denoiser.noise_scale_frame=0",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            env={**os.environ, "HYDRA_FULL_ERROR": "1"},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"RFdiffusion failed (rc={result.returncode}): "
                f"{result.stderr[-500:]}"
            )

        # ProteinMPNN over ALL scaffolds in a single batched call. The
        # ProteinMPNN CLI accepts a directory of PDBs and processes them
        # back-to-back without reloading the model between calls — the GPU
        # stays hot from the first scaffold to the last.
        scaffolds = sorted(tmp_path.glob("design_*.pdb"))
        if not scaffolds:
            raise RuntimeError(
                f"No scaffolds produced; RFdiffusion output dir was empty:\n"
                f"{result.stdout[-300:]}"
            )

        seq_by_path = _run_proteinmpnn_batch(scaffolds, tmp_path / "mpnn_out")

        candidates: list[dict[str, Any]] = []
        for i, scaffold_path in enumerate(scaffolds):
            seq, logprob = seq_by_path.get(scaffold_path.name, ("", -10.0))
            if not seq:
                continue
            candidates.append({
                "candidate_id": f"design_{i:03d}",
                "sequence": seq,
                "length": len(seq),
                "proteinmpnn_logprob": logprob,
                "scaffold_pdb": scaffold_path.read_text(),
                "designed_against_hotspots": hotspot_residues,
                "mock": False,
            })

        candidates.sort(key=lambda c: c["proteinmpnn_logprob"], reverse=True)

    return {
        "candidates": candidates,
        "mock": False,
        "n_designs_requested": n_designs,
    }


def _run_proteinmpnn_batch(
    scaffold_pdbs: "list[Path]",
    out_dir: "Path",
) -> "dict[str, tuple[str, float]]":
    """Run ProteinMPNN on a directory of scaffold PDBs in ONE process.

    ProteinMPNN's helper script `protein_mpnn_run.py` takes a parsed-chain
    JSONL and processes all entries with a single GPU model load. Returns
    `{scaffold_basename: (sequence, mean_logprob)}`.
    """
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    pmpnn_dir = Path(os.environ.get("PROTEINMPNN_DIR", "/opt/ProteinMPNN"))
    parser = pmpnn_dir / "helper_scripts" / "parse_multiple_chains.py"
    runner = pmpnn_dir / "protein_mpnn_run.py"
    if not runner.exists():
        raise RuntimeError(
            f"ProteinMPNN not found at {runner}. Set PROTEINMPNN_DIR or "
            "run scripts/setup_a100.sh."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdbs_dir = out_dir / "pdbs"
    pdbs_dir.mkdir(exist_ok=True)
    for s in scaffold_pdbs:
        shutil.copy(s, pdbs_dir / s.name)

    # 1. Parse all PDBs into a single JSONL
    parsed_jsonl = out_dir / "parsed.jsonl"
    subprocess.run(
        ["python", str(parser),
         "--input_path", str(pdbs_dir),
         "--output_path", str(parsed_jsonl)],
        check=True, capture_output=True,
    )

    # 2. Run ProteinMPNN once over the whole batch
    result = subprocess.run(
        ["python", str(runner),
         "--jsonl_path", str(parsed_jsonl),
         "--out_folder", str(out_dir),
         "--num_seq_per_target", "1",
         "--sampling_temp", "0.1",
         "--seed", "42",
         "--batch_size", "1"],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinMPNN batch run failed (rc={result.returncode}): "
            f"{result.stderr[-500:]}"
        )

    # 3. Parse the per-scaffold FASTAs ProteinMPNN drops in seqs/
    seqs_dir = out_dir / "seqs"
    out: dict[str, tuple[str, float]] = {}
    if not seqs_dir.exists():
        return out
    for fa in seqs_dir.glob("*.fa"):
        # ProteinMPNN's FASTA format: header line includes "score=X.XX"
        text = fa.read_text().splitlines()
        seq_lines: list[str] = []
        last_score = -10.0
        cur_seq: list[str] = []
        for line in text:
            if line.startswith(">"):
                if cur_seq:
                    seq_lines.append("".join(cur_seq))
                    cur_seq = []
                # Pull score field (negative log-prob; we negate to logprob)
                for tok in line.split(","):
                    tok = tok.strip()
                    if tok.startswith("score="):
                        try:
                            last_score = -float(tok.split("=", 1)[1])
                        except ValueError:
                            pass
            else:
                cur_seq.append(line.strip())
        if cur_seq:
            seq_lines.append("".join(cur_seq))
        # Use the last (designed, not native) sequence
        if len(seq_lines) >= 2:
            designed_seq = seq_lines[1]
            out[fa.stem + ".pdb"] = (designed_seq, last_score)
    return out


def _run_proteinmpnn(scaffold_pdb: Path) -> tuple[str, float]:
    """Run ProteinMPNN on a single scaffold PDB and return (sequence, logprob).

    Uses ProteinMPNN's Python API rather than a CLI subprocess so we get
    the per-residue log-probabilities back as floats.
    """
    import os
    import sys
    from pathlib import Path

    pmpnn_dir = os.environ.get(
        "PROTEINMPNN_DIR",
        "/opt/ProteinMPNN",
    )
    if pmpnn_dir not in sys.path:
        sys.path.insert(0, pmpnn_dir)

    try:
        import numpy as np
        import torch
        from protein_mpnn_utils import (             # type: ignore[import-not-found]
            ProteinMPNN, parse_PDB, tied_featurize, _scores,
        )
    except ImportError as e:
        raise RuntimeError(
            f"ProteinMPNN not importable from {pmpnn_dir}. Set PROTEINMPNN_DIR "
            f"or run scripts/setup_a100.sh on the VM. ({e})"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = Path(pmpnn_dir) / "vanilla_model_weights" / "v_48_020.pt"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pdb_dict = parse_PDB([str(scaffold_pdb)], ca_only=False)
    chain_id_dict = {pdb_dict[0]["name"]: ([pdb_dict[0]["chain_id"]], [])}
    fixed_positions_dict = {pdb_dict[0]["name"]: {pdb_dict[0]["chain_id"]: []}}
    omit_AAs_np = np.array([aa in "X" for aa in "ACDEFGHIKLMNPQRSTVWYX"]).astype(np.float32)

    with torch.no_grad():
        feats = tied_featurize(
            pdb_dict, device, chain_id_dict, fixed_positions_dict,
            None, None, None, None, ca_only=False,
        )
        sample_dict = model.sample(
            feats[0], feats[1], feats[2], feats[3], feats[4],
            feats[5], feats[6], feats[7], feats[8],
            randn=torch.randn(feats[0].shape[0], feats[0].shape[1], device=device),
            temperature=0.1, omit_AAs_np=omit_AAs_np, bias_AAs_np=np.zeros(21),
            chain_M_pos=feats[9], omit_AA_mask=feats[10] if len(feats) > 10 else None,
            pssm_coef=None, pssm_bias=None, pssm_multi=0.0,
            pssm_log_odds_flag=False, pssm_log_odds_mask=None,
            pssm_bias_flag=False, bias_by_res=feats[11] if len(feats) > 11 else None,
        )
        S_sample = sample_dict["S"]
        log_probs = model(
            feats[0], S_sample, feats[2], feats[3], feats[4],
            feats[5], feats[6], use_input_decoding_order=True,
            decoding_order=sample_dict["decoding_order"],
        )
        scores = _scores(S_sample, log_probs, feats[6])
        mean_logprob = -scores.mean().item()

    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    seq = "".join(alphabet[i] for i in S_sample[0].cpu().numpy().tolist())
    seq = seq.rstrip("X")
    return seq, mean_logprob
