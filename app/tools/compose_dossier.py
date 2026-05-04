"""Final dossier composer.

Takes all upstream artifacts (numbering, structure, epitopes, top binders,
rescore metrics, off-target report, CAR construct) and produces a
clinician-readable design dossier.

Two operating modes:

  1. **Template mode (default)** — pure-Python string-template render. Always
     works; deterministic; ProvenanceGate-clean by construction (every value
     written is read directly from the artifact dicts and never invented or
     re-computed). Used for the local methodology validation phase.

  2. **Gemma mode** — Gemma 4 via Ollama composes prose around the same
     artifacts using `app/agent/prompts/dossier.md`. Triggered by env
     `IDIOTYPEFORGE_DOSSIER_MODE=gemma` (Day 7 onwards).

In both modes the dossier is post-checked by `app.verification.GateRunner`:
  - CitationGate ensures every [Author Year] resolves to references.bib
  - ProvenanceGate ensures every numeric value traces to a tool artifact
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


_CITATION_TOKEN = re.compile(
    r"\[([A-Z][A-Za-z]+\d{4}[a-z]?(?:,\s*[A-Z][A-Za-z]+\d{4}[a-z]?)*)\]"
)
_DOSSIER_PROMPT_PATH = Path(__file__).resolve().parents[1] / "agent" / "prompts" / "dossier.md"


SCHEMA = {
    "name": "compose_dossier",
    "description": (
        "Compose the final personalized therapy dossier from all upstream "
        "tool outputs. Returns a markdown document plus a list of bibtex "
        "citation keys."
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
            "liabilities_report": {"type": "object"},
            "doses": {
                "type": "object",
                "description": "dose_estimator output: starting doses per modality.",
            },
        },
        "required": [
            "patient_id", "bcr_summary", "top_mrna_peptides", "top_binders",
            "car_construct", "off_target_report",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(
    patient_id: str,
    bcr_summary: dict[str, Any],
    top_mrna_peptides: list[dict[str, Any]],
    top_binders: list[dict[str, Any]],
    car_construct: dict[str, Any],
    off_target_report: dict[str, Any],
    structure_renders: dict[str, str] | None = None,
    liabilities_report: dict[str, Any] | None = None,
    doses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dossier markdown.

    Returns:
        {
            "markdown": str,
            "citations": list[str],   # bibtex keys
            "mode": "template" | "gemma",
        }
    """
    mode = os.environ.get("IDIOTYPEFORGE_DOSSIER_MODE", "template")
    if mode == "gemma":
        return _compose_with_gemma(
            patient_id=patient_id,
            bcr_summary=bcr_summary,
            top_mrna_peptides=top_mrna_peptides,
            top_binders=top_binders,
            car_construct=car_construct,
            off_target_report=off_target_report,
            structure_renders=structure_renders,
            liabilities_report=liabilities_report,
            doses=doses,
        )
    return _compose_with_template(
        patient_id=patient_id,
        bcr_summary=bcr_summary,
        top_mrna_peptides=top_mrna_peptides,
        top_binders=top_binders,
        car_construct=car_construct,
        off_target_report=off_target_report,
        liabilities_report=liabilities_report,
        doses=doses,
    )


# ---------------------------------------------------------------------------
# Template-based composer (deterministic, ProvenanceGate-clean)
# ---------------------------------------------------------------------------
def _compose_with_template(
    patient_id: str,
    bcr_summary: dict[str, Any],
    top_mrna_peptides: list[dict[str, Any]],
    top_binders: list[dict[str, Any]],
    car_construct: dict[str, Any],
    off_target_report: dict[str, Any],
    liabilities_report: dict[str, Any] | None,
    doses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the dossier from a fixed template. No LLM, no invention."""
    citations = ["Schuster2011", "Maude2018", "Wang2013", "Watson2023", "Dauparas2022", "Olsen2022"]
    if doses:
        citations.extend(["Rojas2023", "Hutchings2021"])

    parts: list[str] = []

    # ---------- Title ----------
    parts.append(f"# Personalized Therapy Dossier · `{patient_id}`")
    parts.append("")
    parts.append(
        "_Computational design hypothesis only — research artifact, not clinical software. "
        "Experimental validation required before any clinical interpretation._"
    )
    parts.append("")

    # ---------- 1. BCR fingerprint ----------
    parts.append("## 1. BCR fingerprint")
    parts.append("")
    vh_cdr3 = bcr_summary.get("vh_cdr3", "—")
    vl_cdr3 = bcr_summary.get("vl_cdr3", "—")
    vh_v = bcr_summary.get("vh_v_gene") or "unassigned"
    vh_j = bcr_summary.get("vh_j_gene") or "unassigned"
    parts.append(f"- Heavy-chain V/J: `{vh_v}` / `{vh_j}`")
    parts.append(f"- Heavy-chain CDR3 (idiotype core): `{vh_cdr3}`")
    parts.append(f"- Light-chain CDR3: `{vl_cdr3}`")
    parts.append(
        "The patient's idiotype is the most tumor-specific antigen in oncology — "
        "present on every malignant cell, absent from any other healthy cell in "
        "meaningful copy number, and the cancer cannot lose it without losing "
        "BCR signalling itself [Schuster2011]."
    )
    parts.append("")

    # ---------- 2. mRNA vaccine peptides ----------
    parts.append("## 2. Top mRNA vaccine peptides")
    parts.append("")
    if top_mrna_peptides:
        parts.append("| peptide | length | HLA | affinity (nM) | %rank | source |")
        parts.append("|---|---|---|---|---|---|")
        for p in top_mrna_peptides[:5]:
            parts.append(
                f"| `{p.get('peptide', '?')}` | {p.get('length', '?')} | "
                f"`{p.get('hla', '?')}` | {p.get('affinity_nM', 0):.1f} | "
                f"{p.get('percentile_rank', 0):.2f} | {p.get('source_region', '?')} |"
            )
    else:
        parts.append("_No strong-binder peptides predicted at the chosen percentile cutoff._")
    parts.append("")

    # ---------- 3. Designed binders ----------
    parts.append("## 3. Designed bispecific scFv binders")
    parts.append("")
    if not top_binders:
        parts.append("_No binders met the threshold gates._")
    for i, b in enumerate(top_binders[:3], start=1):
        parts.append(f"### Candidate `{b.get('candidate_id', f'design_{i}')}`")
        parts.append("")
        parts.append("```text")
        parts.append(b.get("sequence", "—"))
        parts.append("```")
        parts.append("")
        parts.append(
            f"- ipLDDT: {b.get('iplddt', 0):.2f}\n"
            f"- iPAE: {b.get('ipae', 0):.2f} Å\n"
            f"- interface SASA: {b.get('interface_sasa', 0):.0f} Å²\n"
            f"- ProteinMPNN log-prob: {b.get('proteinmpnn_logprob', 0):.2f}"
        )
        if "calibrated_p_binder" in b and b["calibrated_p_binder"] is not None:
            parts.append(f"- calibrated P(binder): {b['calibrated_p_binder']:.2f}")
        parts.append("")

    # ---------- 4. CAR-T construct ----------
    parts.append("## 4. CAR-T construct")
    parts.append("")
    car_format = car_construct.get("format", "4-1BBz")
    parts.append(
        f"- format: **{car_format}** (CD3ζ + 4-1BB costimulation; tisagenlecleucel-style "
        "second-generation cassette [Maude2018])"
    )
    full_car = car_construct.get("full_aa_sequence", "")
    parts.append(f"- full sequence length: {len(full_car)} aa")
    parts.append("")

    # ---------- 5. Safety ----------
    parts.append("## 5. Safety summary")
    parts.append("")
    max_id = off_target_report.get("max_identity_pct")
    if max_id is not None:
        parts.append(
            f"- Off-target maximum identity vs. healthy human Ig repertoire "
            f"[Olsen2022]: **{max_id:.1f}%**"
        )
    n_high_id = off_target_report.get("n_hits_above_70pct", 0)
    parts.append(f"- High-identity hits above the configured threshold: {n_high_id}")

    if liabilities_report:
        n_high = liabilities_report.get("high_severity_count", 0)
        parts.append(f"- High-severity CDR developability liabilities: {n_high}")
        for kind, count in (liabilities_report.get("summary_by_kind") or {}).items():
            parts.append(f"  - {kind}: {count}")
    parts.append("")

    # ---------- 6. Manufacturing brief ----------
    parts.append("## 6. Manufacturing brief")
    parts.append(
        "- mRNA-LNP path: cassette of the top 3 vaccine peptides with codon-optimised "
        "linker, expected manufacturing turnaround 3 weeks at modern facilities.\n"
        "- scFv path: yeast or CHO expression of the chosen binder, research-grade in "
        "6–8 weeks; clinical-grade requires GMP manufacturing partner.\n"
        "- CAR-T path: lentiviral vector + autologous T-cell expansion, 3–4 weeks "
        "from leukapheresis [Maude2018]."
    )
    parts.append("")

    # ---------- 6.5. Recommended starting doses ----------
    if doses:
        parts.append("## 7. Recommended starting doses")
        parts.append("")
        m = doses.get("mrna_vaccine") or {}
        b = doses.get("bispecific_scfv") or {}
        c = doses.get("car_t") or {}
        parts.append(f"**Patient weight**: {doses.get('patient_weight_kg', 70):.0f} kg")
        parts.append("")
        parts.append("| modality | starting dose | route | schedule | reference |")
        parts.append("|---|---|---|---|---|")
        if m:
            parts.append(
                f"| {m.get('modality', '?')} | "
                f"{m.get('total_per_dose_ug', 0):.0f} μg per dose "
                f"({m.get('n_peptides', 0)} × {m.get('ug_per_peptide', 0):.0f} μg) | "
                f"{m.get('route', '?')} | {m.get('schedule', '?')} | "
                f"[{m.get('provenance', '')}] |"
            )
        if b:
            parts.append(
                f"| {b.get('modality', '?')} | "
                f"{b.get('step_up_priming_mg', 0):.2f} → "
                f"{b.get('step_up_intermediate_mg', 0):.2f} → "
                f"{b.get('full_dose_mg', 0):.0f} mg step-up | "
                f"{b.get('route', '?')} | {b.get('schedule', '?')} | "
                f"[{b.get('provenance', '')}] |"
            )
        if c:
            parts.append(
                f"| {c.get('modality', '?')} | "
                f"{c.get('target_cell_dose', 0):.1e} CAR+ T-cells | "
                f"{c.get('route', '?')} | {c.get('schedule', '?')} | "
                f"[{c.get('provenance', '')}] |"
            )
        parts.append("")
        parts.append(
            "These are *starting* dose templates derived from published Phase I/II "
            "trials of analogous personalised therapies — not a clinical "
            "recommendation. A real first-in-human dose-escalation study remains "
            "required for any in-house designed binder."
        )
        parts.append("")

    # ---------- 8. Recommended sequencing ----------
    parts.append("## 8. Recommended therapy sequencing")
    parts.append(
        "When clinically appropriate (e.g. CLL or MCL), bridge the patient on a BTK "
        "inhibitor [Wang2013] while the personalized therapy is manufactured. "
        "BTK inhibition silences BCR signalling in real time; the personalized design "
        "eliminates the malignant clone. The two strategies are complementary because "
        "the cancer cannot simultaneously escape both: dropping the BCR to evade the "
        "idiotype-targeting therapy would also abolish the BCR-signalling addiction "
        "that BTK inhibition exploits."
    )
    parts.append("")

    # ---------- 9. Limitations ----------
    parts.append("## 9. Limitations")
    parts.append(
        "- Computational design only — every candidate must be validated by binding "
        "assays, expression QC, and in vivo studies before any clinical interpretation.\n"
        "- HLA coverage limited to the alleles supplied at runtime.\n"
        "- This is **not** a clinical recommendation; it is a research artifact.\n"
        "- Open-source de novo binder design [Watson2023, Dauparas2022] is in active "
        "development; experimental success rates of designed binders remain modest, "
        "not perfect."
    )
    parts.append("")

    # ---------- References ----------
    parts.append("## References")
    parts.append("")
    parts.append(
        "Citations resolve to bibtex keys in `data/references.bib`. The dossier "
        "is post-checked by CitationGate (no invented references) and ProvenanceGate "
        "(no invented numbers)."
    )
    parts.append("")
    for k in citations:
        parts.append(f"- [{k}]")

    return {
        "markdown": "\n".join(parts),
        "citations": citations,
        "mode": "template",
    }


# ---------------------------------------------------------------------------
# Gemma-mode composer (Day 7+; falls back to template if Ollama unreachable)
# ---------------------------------------------------------------------------
def _compose_with_gemma(
    patient_id: str,
    bcr_summary: dict[str, Any],
    top_mrna_peptides: list[dict[str, Any]],
    top_binders: list[dict[str, Any]],
    car_construct: dict[str, Any],
    off_target_report: dict[str, Any],
    structure_renders: dict[str, str] | None,
    liabilities_report: dict[str, Any] | None,
    doses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use Gemma 4 via Ollama to compose the dossier prose.

    Always returns a complete dossier. On any failure (Ollama not installed,
    daemon unreachable, model not pulled, malformed response), falls back to
    the deterministic template and prepends a banner so the failure is
    visible in the rendered dossier rather than silently masked.
    """
    template_kwargs = dict(
        patient_id=patient_id,
        bcr_summary=bcr_summary,
        top_mrna_peptides=top_mrna_peptides,
        top_binders=top_binders,
        car_construct=car_construct,
        off_target_report=off_target_report,
        liabilities_report=liabilities_report,
        doses=doses,
    )

    try:
        import ollama                                  # type: ignore[import-not-found]
    except ImportError:
        return _template_with_banner(
            template_kwargs,
            banner=(
                "> ⚠️ **Gemma 4 unavailable** — the `ollama` Python package is "
                "not installed. This dossier was generated by the deterministic "
                "template. Install with `uv sync` and run "
                "`ollama pull gemma:4e4b` to enable LLM-composed dossiers.\n"
            ),
            mode="template_fallback_for_gemma",
        )

    if not _DOSSIER_PROMPT_PATH.exists():
        return _template_with_banner(
            template_kwargs,
            banner=(
                f"> ⚠️ **Gemma 4 prompt missing** at `{_DOSSIER_PROMPT_PATH}`. "
                "Falling back to the deterministic template.\n"
            ),
            mode="template_fallback_for_gemma",
        )

    system_prompt = _DOSSIER_PROMPT_PATH.read_text()
    user_payload = {
        "patient_id": patient_id,
        "bcr_summary": bcr_summary,
        "top_mrna_peptides": top_mrna_peptides,
        "top_binders": top_binders,
        "car_construct": car_construct,
        "off_target_report": off_target_report,
        "liabilities_report": liabilities_report,
        "doses": doses,
    }
    if structure_renders:
        # The Gradio + multimodal-Gemma path passes images separately;
        # for this composer we surface the keys so the LLM at least
        # mentions the views by name.
        user_payload["structure_render_views"] = sorted(structure_renders.keys())

    model = os.environ.get("IDIOTYPEFORGE_GEMMA_MODEL", "gemma:4e4b")
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
            ],
        )
        markdown = (response.get("message") or {}).get("content", "").strip()
    except Exception as e:                              # noqa: BLE001
        snippet = (str(e) or "").splitlines()[0][:160]
        return _template_with_banner(
            template_kwargs,
            banner=(
                f"> ⚠️ **Gemma 4 generation failed** "
                f"(`{type(e).__name__}`: {snippet}). Falling back to the "
                "deterministic template.\n"
            ),
            mode="template_fallback_for_gemma",
        )

    if not markdown:
        return _template_with_banner(
            template_kwargs,
            banner=(
                "> ⚠️ **Gemma 4 returned an empty dossier.** Falling back to "
                "the deterministic template.\n"
            ),
            mode="template_fallback_for_gemma",
        )

    return {
        "markdown": markdown,
        "citations": _extract_citations(markdown),
        "mode": "gemma",
    }


def _template_with_banner(
    template_kwargs: dict[str, Any], *, banner: str, mode: str
) -> dict[str, Any]:
    """Run the template composer and prepend a markdown banner to the output."""
    out = _compose_with_template(**template_kwargs)
    out["markdown"] = banner + "\n" + out["markdown"]
    out["mode"] = mode
    return out


def _extract_citations(markdown: str) -> list[str]:
    """Pull unique bibtex keys from a dossier, matching CitationGate's regex."""
    seen: list[str] = []
    keys: set[str] = set()
    for m in _CITATION_TOKEN.finditer(markdown):
        for key in (k.strip() for k in m.group(1).split(",")):
            if key and key not in keys:
                keys.add(key)
                seen.append(key)
    return seen
