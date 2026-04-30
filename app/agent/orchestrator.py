"""Gemma 4 agent loop.

Takes a patient BCR (VH + VL + optional HLA) and runs the full design
pipeline by letting Gemma 4 call the registered tools natively. Streams
tool-call events for the UI.

Default model: `gemma:4e4b` via Ollama HTTP. Swap to `gemma:4-26b` for the
heavier dossier-synthesis pass when GPU is available.

CLI:
    python -m app.agent.orchestrator \\
        --vh-fasta data/demo_cases/fl_carlotti2009.fasta \\
        --hla "HLA-A*02:01" \\
        --out runs/fl_001/
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import ollama

from app.agent.router import dispatch, gemma_tool_specs
from app.verification import (
    ArtifactStore,
    CitationGate,
    GateRunner,
    MockModeGate,
    ProvenanceGate,
    ThresholdGate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("IDIOTYPEFORGE_GEMMA_MODEL", "gemma:4e4b")
MAX_AGENT_STEPS = int(os.environ.get("IDIOTYPEFORGE_MAX_STEPS", "20"))


@dataclass
class AgentEvent:
    """One streamed event for the UI tool-call log."""

    kind: str   # "thought", "tool_call", "tool_result", "final"
    payload: Any
    timestamp: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class PatientInput:
    patient_id: str
    vh_sequence: str
    vl_sequence: str
    hla_alleles: list[str]


# ---------------------------------------------------------------------------
# Provenance-aware tool dispatch
# ---------------------------------------------------------------------------
def dispatch_traced(
    name: str,
    args: dict[str, Any],
    store: ArtifactStore,
) -> dict[str, Any]:
    """Wrap router.dispatch and record the call into the artifact store."""
    output = dispatch(name, args)
    store.record(tool_name=name, args=args, output=output)
    return output


# ---------------------------------------------------------------------------
# Final verification stage
# ---------------------------------------------------------------------------
def verify_dossier(
    dossier_markdown: str,
    store: ArtifactStore,
    abort_on: str = "error",
) -> dict[str, Any]:
    """Run the verification gate pipeline against the composed dossier.

    Returns:
        {
            "passed": bool,
            "audit_markdown": str,
            "results": [GateResult.__dict__, ...],
        }
    """
    runner = GateRunner(abort_on=abort_on)
    overall, results = runner.run([
        (MockModeGate(),     {"store": store}),
        (CitationGate(),     {"dossier_markdown": dossier_markdown}),
        (ProvenanceGate(),   {"dossier_markdown": dossier_markdown, "store": store}),
    ])
    return {
        "passed": overall,
        "audit_markdown": runner.report_markdown(results),
        "results": [r.__dict__ for r in results],
    }


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------
def _load_system_prompt() -> str:
    p = Path(__file__).parent / "prompts" / "system.md"
    if not p.exists():
        return "You are IdiotypeForge, a personalized lymphoma therapy designer."
    return p.read_text()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(
    patient: PatientInput,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_AGENT_STEPS,
) -> Iterator[AgentEvent]:
    """Run the agent loop and yield events. The final event has kind='final'.

    Day-7 implementation:
        - construct messages = [system, user(patient JSON)]
        - loop: call ollama.chat(model, messages, tools=gemma_tool_specs())
            - if response.tool_calls: dispatch each, append result, continue
            - else: emit final, break
        - hard cap at max_steps
    """
    yield AgentEvent("thought", f"agent starting for patient {patient.patient_id}")

    # Stub: emit a placeholder final event so the UI flow is exercisable now.
    yield AgentEvent(
        "final",
        {
            "patient_id": patient.patient_id,
            "status": "stub",
            "message": (
                "orchestrator stub — implement on Day 7 once tool wrappers are "
                "live. The UI and event-streaming contract is already wired."
            ),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--vh", "vh_fasta", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--vl", "vl_fasta", type=click.Path(exists=True, dir_okay=False))
@click.option("--hla", "hla", default="HLA-A*02:01", help="Comma-separated HLA-I alleles.")
@click.option("--patient-id", default="demo")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="runs/")
@click.option("--model", default=DEFAULT_MODEL)
def cli(vh_fasta: str, vl_fasta: str | None, hla: str, patient_id: str, out_dir: str, model: str) -> None:
    """Run IdiotypeForge end-to-end on a single patient BCR."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    vh_seq = _read_fasta_first_record(Path(vh_fasta))
    vl_seq = _read_fasta_first_record(Path(vl_fasta)) if vl_fasta else ""

    patient = PatientInput(
        patient_id=patient_id,
        vh_sequence=vh_seq,
        vl_sequence=vl_seq,
        hla_alleles=[a.strip() for a in hla.split(",")],
    )

    out = Path(out_dir) / patient_id
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / "events.jsonl"

    with events_path.open("w") as fh:
        for ev in run_agent(patient, model=model):
            line = json.dumps({"kind": ev.kind, "ts": ev.timestamp, "payload": ev.payload}, default=str)
            fh.write(line + "\n")
            click.echo(line)

    click.echo(f"\nDone. Events log: {events_path}")


def _read_fasta_first_record(p: Path) -> str:
    seq = []
    with p.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line.strip())
    return "".join(seq)


if __name__ == "__main__":
    cli()
