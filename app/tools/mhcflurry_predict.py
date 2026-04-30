"""MHC class-I epitope prediction over CDR3 peptides.

Uses MHCflurry 2.x (O'Donnell 2020). Fully CPU-runnable. Slides 8–11-mer
windows across CDR3-H and CDR3-L, scores against the patient's HLA-I
alleles, and returns the top-K predicted binders for an mRNA vaccine cassette.

Lazy import: the heavy `mhcflurry` import + ~150 MB model load happens
inside `_load_predictor()`. The module imports fast even without MHCflurry.

First-time setup (one-shot):
    mhcflurry-downloads fetch models_class1_presentation

Verification target (per plan §4):
    Reproduce 3 published HLA-A*02:01 epitopes within 0.5-log IC50.

Reference: O'Donnell et al. 2020 *Cell Systems* (ODonnell2020).
"""
from __future__ import annotations

import functools
from typing import Any


SCHEMA = {
    "name": "predict_mhc_epitopes",
    "description": (
        "Predict HLA class-I-restricted peptides derived from the patient's "
        "CDR3 loops. Used to design the personalized mRNA vaccine cassette."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cdr3_h_aa": {"type": "string"},
            "cdr3_l_aa": {"type": "string"},
            "hla_alleles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patient HLA-I alleles, e.g. ['HLA-A*02:01', 'HLA-B*07:02'].",
            },
            "lengths": {
                "type": "array",
                "items": {"type": "integer"},
                "default": [8, 9, 10, 11],
            },
            "top_k": {"type": "integer", "default": 10},
            "percentile_cutoff": {
                "type": "number",
                "default": 2.0,
                "description": "Keep peptides with %rank ≤ cutoff (lower = stronger binder).",
            },
        },
        "required": ["cdr3_h_aa", "cdr3_l_aa", "hla_alleles"],
    },
}


# ---------------------------------------------------------------------------
# Window enumeration
# ---------------------------------------------------------------------------
def slide_windows(seq: str, lengths: list[int]) -> list[str]:
    """All k-mers of the requested lengths, deduplicated, in order."""
    seen: set[str] = set()
    out: list[str] = []
    for k in lengths:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in seen:
                continue
            # Skip kmers with non-canonical AAs (X, *, gaps)
            if any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in kmer):
                continue
            seen.add(kmer)
            out.append(kmer)
    return out


# ---------------------------------------------------------------------------
# Predictor (cached load)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _load_predictor() -> Any:
    """Lazy-load the MHCflurry predictor; cached across calls."""
    try:
        from mhcflurry import Class1AffinityPredictor  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "MHCflurry is not installed. Install with `uv pip install mhcflurry` "
            "and then download the models with: "
            "`mhcflurry-downloads fetch models_class1_presentation`."
        ) from e
    try:
        return Class1AffinityPredictor.load()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "MHCflurry models not found. Run: "
            "`mhcflurry-downloads fetch models_class1_presentation` "
            "(one-time download, ~150 MB)."
        ) from e


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(
    cdr3_h_aa: str,
    cdr3_l_aa: str,
    hla_alleles: list[str],
    lengths: list[int] | None = None,
    top_k: int = 10,
    percentile_cutoff: float = 2.0,
) -> dict[str, Any]:
    """Slide windows over CDR3 sequences, score with MHCflurry, return top-K.

    Returns:
        {
            "epitopes": [
                {peptide, hla, affinity_nM, percentile_rank, length, source_region}, ...
            ],
            "n_evaluated": int,
            "n_strong_binders": int,
        }
    """
    cdr3_h = cdr3_h_aa.strip().upper()
    cdr3_l = cdr3_l_aa.strip().upper()
    if not cdr3_h or not cdr3_l:
        raise ValueError("Both cdr3_h_aa and cdr3_l_aa are required.")
    if not hla_alleles:
        raise ValueError("Provide at least one HLA-I allele.")

    lengths_ = lengths or [8, 9, 10, 11]

    h_kmers = slide_windows(cdr3_h, lengths_)
    l_kmers = slide_windows(cdr3_l, lengths_)
    all_kmers = h_kmers + l_kmers
    region_of: dict[str, str] = {**{k: "CDR3-H" for k in h_kmers}, **{k: "CDR3-L" for k in l_kmers}}

    if not all_kmers:
        return {"epitopes": [], "n_evaluated": 0, "n_strong_binders": 0}

    predictor = _load_predictor()

    # MHCflurry prefers a flat (peptide, allele) grid; expand the cross-product.
    peptides_grid = []
    alleles_grid = []
    for kmer in all_kmers:
        for hla in hla_alleles:
            peptides_grid.append(kmer)
            alleles_grid.append(hla)

    df = predictor.predict_to_dataframe(peptides=peptides_grid, alleles=alleles_grid)

    epitopes: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rank_value = row.get("mhcflurry_affinity_percentile")
        try:
            rank_float = float(rank_value)
        except (TypeError, ValueError):
            continue
        if rank_float > percentile_cutoff:
            continue
        peptide = str(row["peptide"])
        epitopes.append({
            "peptide": peptide,
            "hla": str(row["allele"]),
            "affinity_nM": float(row.get("mhcflurry_affinity", float("inf"))),
            "percentile_rank": rank_float,
            "length": len(peptide),
            "source_region": region_of.get(peptide, "CDR3-H"),
        })

    epitopes.sort(key=lambda e: e["affinity_nM"])
    top = epitopes[:top_k]

    return {
        "epitopes": top,
        "n_evaluated": len(peptides_grid),
        "n_strong_binders": len(epitopes),
    }
