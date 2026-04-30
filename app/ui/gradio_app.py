"""Gradio dashboard for IdiotypeForge.

Layout:
    [Top]    Title bar with hackathon badge + brief tagline
    [Left]   BCR input (VH textarea, VL textarea, HLA dropdown) +
             3 "Load example case" buttons (FL, CLL, DLBCL)
    [Right]  Streaming agent log (tool calls + rationales) +
             Decision-card panel (top-3 binders) +
             Verification audit panel +
             Dossier markdown render
    [Footer] "Research artifact, not clinical software" disclaimer

Deployed to HF Space ZeroGPU on Day 15. Three demo cases load instantly
from the IMGT-germline-derived fixtures in `data/demo_cases/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from app.agent.orchestrator import PatientInput, run_agent
from app.ui.decision_card import render_card

DEMO_DIR = Path(__file__).parent.parent.parent / "data" / "demo_cases"


# ---------------------------------------------------------------------------
# Demo case loaders
# ---------------------------------------------------------------------------
DEMO_CASES = {
    "FL — IGHV4-34 (Carlotti 2009)": "fl_carlotti2009",
    "CLL — Subset #2 (Stamatopoulos 2017)": "cll_subset2",
    "DLBCL — GCB IGHV3-23 (Young 2015)": "dlbcl_young2015",
}


def _load_demo_case(case_id: str) -> tuple[str, str, str, str]:
    """Read VH + VL records from a demo FASTA. Returns (vh, vl, hla_default, patient_id)."""
    fa = DEMO_DIR / f"{case_id}.fasta"
    if not fa.exists():
        return ("", "", "HLA-A*02:01", case_id)
    vh, vl = "", ""
    cur = None
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            cur = "VH" if "|VH" in line.upper() else "VL"
            continue
        if cur == "VH":
            vh += line.strip()
        elif cur == "VL":
            vl += line.strip()
    return vh, vl, "HLA-A*02:01", case_id


# ---------------------------------------------------------------------------
# Event rendering
# ---------------------------------------------------------------------------
_KIND_ICONS = {
    "thought":     "💭",
    "tool_call":   "🔧",
    "tool_result": "  ✓",
    "verification": "🛡️",
    "final":       "✅",
    "error":       "❌",
}


def _format_event(kind: str, payload: Any) -> str:
    icon = _KIND_ICONS.get(kind, "·")
    if kind == "thought":
        return f"\n{icon} **{payload}**\n"
    if kind == "tool_call" and isinstance(payload, dict):
        name = payload.get("name", "?")
        rationale = payload.get("rationale", "")
        return f"{icon} `{name}()` — *{rationale}*\n"
    if kind == "tool_result" and isinstance(payload, dict):
        name = payload.get("name", "?")
        result = payload.get("result", "")
        result_str = result if isinstance(result, str) else json.dumps(result, default=str)
        preview = result_str[:120] + ("…" if len(result_str) > 120 else "")
        return f"{icon} _{name}_ → `{preview}`\n"
    if kind == "verification":
        if isinstance(payload, dict):
            audit = payload.get("audit_markdown", "")
            return f"\n{icon} **Verification**\n{audit}\n"
        return f"{icon} {payload}\n"
    if kind == "final":
        return f"\n{icon} **Done.**\n"
    if kind == "error":
        return f"{icon} **Error**: {payload}\n"
    return f"  · {payload}\n"


# ---------------------------------------------------------------------------
# Decision-card extraction from final payload
# ---------------------------------------------------------------------------
def _build_decision_cards(final_payload: dict[str, Any]) -> str:
    """Render the top-3 binder decision cards from the final dossier markdown.

    Right now we extract this from the agent events' tool_results in a future
    iteration. For now: parse the dossier markdown's "Designed bispecific scFv
    binders" section into a Markdown block of decision cards.
    """
    md = final_payload.get("dossier_markdown", "")
    if "## 3. Designed bispecific scFv binders" not in md:
        return "_No decision cards available._"
    section = md.split("## 3. Designed bispecific scFv binders", 1)[1]
    section = section.split("## 4.", 1)[0]
    return "## Decision cards (top-3 binders)\n" + section.strip()


# ---------------------------------------------------------------------------
# Audit pretty-print
# ---------------------------------------------------------------------------
def _format_audit(final_payload: dict[str, Any]) -> str:
    audit = final_payload.get("audit_markdown", "")
    passed = final_payload.get("verification_passed")
    badge = "✅ **PASSED**" if passed else "❌ **FAILED**"
    return f"### Verification status: {badge}\n\n{audit}"


# ---------------------------------------------------------------------------
# Pipeline runner adapted to Gradio's incremental output
# ---------------------------------------------------------------------------
def run_pipeline(
    vh: str,
    vl: str,
    hla: str,
    patient_id: str,
    mode: str,
) -> Iterator[tuple[str, str, str, str]]:
    """Yield (log_md, audit_md, cards_md, dossier_md) progressively."""
    if not vh or not vl:
        yield ("⚠️ Both VH and VL sequences are required.", "", "", "")
        return

    log_md = ""
    audit_md = "_Verification will run after the dossier is composed…_"
    cards_md = "_Designed binders will appear here…_"
    dossier_md = "_Working…_"

    patient = PatientInput(
        patient_id=patient_id or "demo",
        vh_sequence=vh.strip(),
        vl_sequence=vl.strip(),
        hla_alleles=[a.strip() for a in hla.split(",") if a.strip()],
    )

    for ev in run_agent(patient, mode=mode):
        log_md += _format_event(ev.kind, ev.payload)
        if ev.kind == "final" and isinstance(ev.payload, dict):
            audit_md = _format_audit(ev.payload)
            cards_md = _build_decision_cards(ev.payload)
            dossier_md = ev.payload.get("dossier_markdown", "")
            yield (log_md, audit_md, cards_md, dossier_md)
            return
        yield (log_md, audit_md, cards_md, dossier_md)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
HEADER_MD = """\
# 🧬 IdiotypeForge

### From a tumour BCR to a personalised therapy dossier — in hours, not months.

*Open-source · CC-BY 4.0 · Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)*
"""

FOOTER_MD = """\
---
**Research artifact, not clinical software.** No patient data is stored or
transmitted. All demonstrated sequences are sourced from public peer-reviewed
publications. Designed binders are computational hypotheses; experimental
validation is required before any clinical interpretation.

Built on Gemma 4 · IgFold · RFdiffusion · ProteinMPNN · MHCflurry · ANARCI · Observed Antibody Space.
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="IdiotypeForge — personalised lymphoma therapy designer", theme=gr.themes.Soft()) as app:
        gr.Markdown(HEADER_MD)

        with gr.Row():
            # ─── LEFT COLUMN ─── input + demo cases
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 1. Patient input")
                vh = gr.Textbox(label="VH sequence", lines=4,
                                placeholder="EVQLVQSGGGLVKPGG…",
                                info="Heavy-chain variable region (~120 aa)")
                vl = gr.Textbox(label="VL sequence", lines=4,
                                placeholder="DIQMTQSPSSLSAS…",
                                info="Light-chain variable region (~110 aa)")
                hla = gr.Textbox(label="Patient HLA-I",
                                 value="HLA-A*02:01",
                                 info="Comma-separate multiple alleles")
                patient_id = gr.Textbox(label="Patient ID", value="demo")

                gr.Markdown("**Or load a published demo case:**")
                with gr.Row():
                    for label, case_id in DEMO_CASES.items():
                        btn = gr.Button(label, size="sm")
                        btn.click(
                            fn=lambda cid=case_id: _load_demo_case(cid),
                            outputs=[vh, vl, hla, patient_id],
                        )

                mode = gr.Radio(
                    label="Agent mode",
                    choices=["template", "gemma"],
                    value="template",
                    info="`template` = deterministic pipeline (no LLM). "
                         "`gemma` = Gemma 4 via Ollama.",
                )

                run_btn = gr.Button("⚡  Design therapy", variant="primary", size="lg")

                gr.Markdown(
                    "**What will run:**\n"
                    "1. ANARCI numbering (CDR1/2/3)\n"
                    "2. IgFold structure prediction\n"
                    "3. CDR liability scan\n"
                    "4. MHCflurry epitope prediction\n"
                    "5. RFdiffusion + ProteinMPNN binder design\n"
                    "6. AlphaFold-Multimer rescore\n"
                    "7. Off-target safety (OAS + UniProt)\n"
                    "8. CAR-T cassette assembly\n"
                    "9. Dossier composition + verification gates"
                )

            # ─── RIGHT COLUMN ─── live log + outputs
            with gr.Column(scale=4, min_width=420):
                with gr.Tabs():
                    with gr.Tab("🔁  Agent activity"):
                        log = gr.Markdown(value="_Awaiting input. Pick a demo case or paste a BCR sequence._",
                                          label="Streaming tool-call log")

                    with gr.Tab("🛡️  Verification audit"):
                        audit = gr.Markdown(value="_The verification audit will appear here once the dossier is composed._")

                    with gr.Tab("🧪  Decision cards"):
                        cards = gr.Markdown(value="_Top-3 designed binders will appear here as ranked decision cards._")

                    with gr.Tab("📋  Therapy dossier"):
                        dossier = gr.Markdown(value="_The full markdown dossier will render here._")

        run_btn.click(
            fn=run_pipeline,
            inputs=[vh, vl, hla, patient_id, mode],
            outputs=[log, audit, cards, dossier],
        )

        gr.Markdown(FOOTER_MD)

    return app


def main() -> None:
    # Sensible defaults: mocks ON locally, template-mode agent.
    os.environ.setdefault("IDIOTYPEFORGE_USE_MOCKS", "1")
    os.environ.setdefault("IDIOTYPEFORGE_AGENT_MODE", "template")
    build_app().queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
