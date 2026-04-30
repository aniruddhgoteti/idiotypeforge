"""MHC class-I epitope prediction over CDR3 peptides.

Uses MHCflurry 2.x (Apache 2.0). Fully CPU-runnable. Slides 8–11-mer windows
across CDR3-H and CDR3-L, scores against the patient HLA alleles, and returns
the top-K predicted binders for an mRNA vaccine cassette.

Verification target (per plan §4):
    Reproduce 3 published HLA-A*02:01 epitopes within 0.5-log IC50.
"""
from __future__ import annotations

from typing import Any

from ._types import MHCEpitope


SCHEMA = {
    "name": "predict_mhc_epitopes",
    "description": (
        "Predict HLA class-I-restricted peptides derived from the patient's CDR3 "
        "loops. Used to design the personalized mRNA vaccine cassette."
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
        },
        "required": ["cdr3_h_aa", "cdr3_l_aa", "hla_alleles"],
    },
}


def run(
    cdr3_h_aa: str,
    cdr3_l_aa: str,
    hla_alleles: list[str],
    lengths: list[int] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Slide windows over CDR3, score with MHCflurry, return top-K epitopes.

    Day-4 implementation:
        - mhcflurry.Class1AffinityPredictor.load() (downloads ~150 MB once)
        - generate all peptides of `lengths` from CDR3-H and CDR3-L
        - predictor.predict_to_dataframe(peptides, alleles)
        - keep peptides w/ percentile_rank <= 2.0 (strong binders)
        - sort by affinity, return top_k as list[MHCEpitope]
    """
    raise NotImplementedError(
        "Stub: implement on Day 4. "
        "First-time install: `mhcflurry-downloads fetch models_class1_presentation`."
    )
