"""Final dossier composer.

Takes all upstream artifacts (numbering, structure, epitopes, top binders,
rescore metrics, off-target report, CAR construct) and asks Gemma 4 to write
a clinician-readable design dossier with:

  - Patient summary
  - Top-3 mRNA vaccine peptides (table + per-peptide rationale)
  - Top-3 designed bispecific scFv binders (decision cards)
  - CAR-T construct cassette
  - Off-target safety summary
  - Manufacturing brief
  - Recommended sequencing protocol (combine with ibrutinib pre-treatment)

Citation discipline: the dossier prompt instructs Gemma 4 that it may ONLY
cite from `data/references.bib`. A regex post-check rejects any free-text
citations.
"""
from __future__ import annotations

from typing import Any


SCHEMA = {
    "name": "compose_dossier",
    "description": (
        "Compose the final personalized therapy dossier from all upstream tool "
        "outputs. Returns a markdown document plus a list of bibtex citation keys."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {"type": "string"},
            "bcr_summary": {"type": "object"},
            "top_mrna_peptides": {"type": "array"},
            "top_binders": {"type": "array"},
            "car_construct": {"type": "object"},
            "off_target_report": {"type": "object"},
            "structure_renders": {
                "type": "object",
                "description": "{view_name: png_b64} for multimodal grounding.",
            },
        },
        "required": [
            "patient_id", "bcr_summary", "top_mrna_peptides", "top_binders",
            "car_construct", "off_target_report",
        ],
    },
}


def run(
    patient_id: str,
    bcr_summary: dict[str, Any],
    top_mrna_peptides: list[dict[str, Any]],
    top_binders: list[dict[str, Any]],
    car_construct: dict[str, Any],
    off_target_report: dict[str, Any],
    structure_renders: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the dossier markdown by calling Gemma 4 with the dossier prompt.

    Day-8 implementation:
        - read app/agent/prompts/dossier.md
        - inject all artifact JSON + structure renders as multimodal content
        - call Gemma 4 (Ollama HTTP) with `format=markdown`
        - regex-check every `[Author Year]` citation resolves to references.bib
        - render markdown -> PDF via weasyprint for the download button
    """
    raise NotImplementedError("Stub: implement on Day 8.")
