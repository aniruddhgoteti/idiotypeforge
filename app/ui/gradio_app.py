"""Gradio UI for IdiotypeForge.

Layout:
    [Top]    Title bar with hackathon badge + benchmarks ribbon
    [Left]   BCR input (VH textarea, VL textarea, HLA dropdown) +
             3 "Load example case" buttons (FL, CLL, DLBCL)
    [Right]  Streaming agent log (tool calls + rationales) +
             Decision-card panel (top-3 binders) +
             Dossier markdown render + Download PDF / PDB buttons
    [Footer] "Research artifact, not clinical software" disclaimer + repo link

Deployed to HF Space ZeroGPU on Day 15. Three demo cases load instantly from
the cached fixtures in `data/demo_cases/<id>/dossier.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from app.agent.orchestrator import PatientInput, run_agent

DEMO_DIR = Path(__file__).parent.parent.parent / "data" / "demo_cases"


# ---------------------------------------------------------------------------
# Demo case loaders
# ---------------------------------------------------------------------------
DEMO_CASES = {
    "FL — Carlotti 2009": "fl_carlotti2009",
    "CLL — Subset #2 (Stamatopoulos 2017)": "cll_subset2",
    "DLBCL — Young 2015 (GCB)": "dlbcl_young2015",
}


def _load_demo_case(case_id: str) -> tuple[str, str, str]:
    """Read the demo FASTA and return (vh_seq, vl_seq, hla_default)."""
    fa = DEMO_DIR / f"{case_id}.fasta"
    if not fa.exists():
        return ("", "", "HLA-A*02:01")
    vh, vl = "", ""
    cur = None
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            cur = "VH" if "VH" in line.upper() else "VL"
            continue
        if cur == "VH":
            vh += line.strip()
        elif cur == "VL":
            vl += line.strip()
    return vh, vl, "HLA-A*02:01"


def _load_cached_dossier(case_id: str) -> dict[str, Any] | None:
    p = DEMO_DIR / case_id / "dossier.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Agent → UI event stream adapter
# ---------------------------------------------------------------------------
def _format_event(event_kind: str, payload: Any) -> str:
    if event_kind == "thought":
        return f"💭 **{payload}**\n"
    if event_kind == "tool_call":
        name = payload.get("name", "?")
        rationale = payload.get("rationale", "")
        return f"🔧 `{name}()` — *{rationale}*\n"
    if event_kind == "tool_result":
        name = payload.get("name", "?")
        return f"  ✓ {name} returned {len(json.dumps(payload.get('result', {})))} bytes\n"
    if event_kind == "final":
        return f"\n✅ **Done.** {payload}\n"
    return f"  · {payload}\n"


def run_pipeline(
    vh: str,
    vl: str,
    hla: str,
    patient_id: str,
) -> Iterator[tuple[str, str]]:
    """Yield (log_md, dossier_md) progressively as the agent runs."""
    if not vh or not vl:
        yield ("⚠️ Both VH and VL sequences are required.", "")
        return

    log_md = ""
    patient = PatientInput(
        patient_id=patient_id or "demo",
        vh_sequence=vh.strip(),
        vl_sequence=vl.strip(),
        hla_alleles=[a.strip() for a in hla.split(",") if a.strip()],
    )
    for ev in run_agent(patient):
        log_md += _format_event(ev.kind, ev.payload)
        if ev.kind == "final":
            dossier = ev.payload.get("dossier_markdown") if isinstance(ev.payload, dict) else ""
            yield (log_md, dossier or "_Dossier composition pending — see logs._")
            return
        yield (log_md, "_Working…_")


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="IdiotypeForge — Personalized lymphoma therapy designer") as app:
        gr.Markdown(
            """
            # 🧬 IdiotypeForge
            ### From tumor sample to personalized therapy dossier in hours.
            *Kaggle Gemma 4 Good Hackathon · open-source under CC-BY 4.0*
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Input")
                vh = gr.Textbox(label="VH sequence", lines=4, placeholder="EVQLVQSGGG...")
                vl = gr.Textbox(label="VL sequence", lines=4, placeholder="DIQMTQSPSS...")
                hla = gr.Textbox(label="Patient HLA-I", value="HLA-A*02:01")
                patient_id = gr.Textbox(label="Patient ID", value="demo")

                gr.Markdown("**Or load a published case:**")
                with gr.Row():
                    for label, case_id in DEMO_CASES.items():
                        btn = gr.Button(label, size="sm")
                        btn.click(
                            fn=lambda cid=case_id: _load_demo_case(cid),
                            outputs=[vh, vl, hla],
                        )

                run_btn = gr.Button("⚡ Design therapy", variant="primary", size="lg")

            with gr.Column(scale=3):
                gr.Markdown("### Agent activity")
                log = gr.Markdown(value="_Awaiting input…_")
                gr.Markdown("### Therapy dossier")
                dossier = gr.Markdown(value="")

        run_btn.click(
            fn=run_pipeline,
            inputs=[vh, vl, hla, patient_id],
            outputs=[log, dossier],
        )

        gr.Markdown(
            """
            ---
            **Research artifact, not clinical software.** No patient data is used; all
            demonstrated sequences are sourced from public peer-reviewed publications.
            Designed binders are computational hypotheses; experimental validation is
            required before any clinical interpretation.

            Built on Gemma 4 · IgFold · RFdiffusion · ProteinMPNN · MHCflurry · Observed Antibody Space.
            """
        )

    return app


def main() -> None:
    build_app().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
