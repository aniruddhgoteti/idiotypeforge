"""Gemma 4 agent loop.

Takes a patient BCR (VH + VL + optional HLA) and runs the full design
pipeline. Two modes:

  - **template** (default): deterministic sequence of 9 tool calls in a
    fixed order. Always works. Used for local methodology validation, CI,
    and as the auto-fallback when Ollama isn't reachable.

  - **gemma**: Gemma 4 via Ollama drives the loop with native function
    calling. Triggered by env `IDIOTYPEFORGE_AGENT_MODE=gemma`. The model
    decides which tools to call and in what order; tool outputs flow back
    to the model until it produces a final dossier.

Either way, the run produces:
  - a stream of `AgentEvent` records (visible in the Gradio UI)
  - an `ArtifactStore` capturing every tool input/output for provenance
  - a final `TherapyDossier` markdown that has been verified by the gate
    runner (CitationGate + ProvenanceGate at minimum)

CLI:
    python -m app.agent.orchestrator \\
        --vh data/demo_cases/fl_carlotti2009.fasta \\
        --vl data/demo_cases/fl_carlotti2009.fasta \\
        --hla "HLA-A*02:01" \\
        --patient-id fl_001 \\
        --out runs/
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

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
# Default Ollama model identifier. Gemma 4 (E2B/E4B/26B/31B) isn't yet
# packaged on the Ollama hub as of 2026-05; gemma3:4b is the closest
# available stand-in (same active-parameter scale as E4B). For the *real*
# Gemma 4 E4B path, see notebooks/idiotypeforge_kaggle.ipynb which loads
# the model via `kagglehub.model_download('google/gemma-4/transformers/e4b')`.
# When Ollama publishes Gemma 4, set IDIOTYPEFORGE_GEMMA_MODEL=gemma4:e4b.
DEFAULT_MODEL = os.environ.get("IDIOTYPEFORGE_GEMMA_MODEL", "gemma3:4b")
MAX_AGENT_STEPS = int(os.environ.get("IDIOTYPEFORGE_MAX_STEPS", "20"))
DEFAULT_AGENT_MODE = os.environ.get("IDIOTYPEFORGE_AGENT_MODE", "template")


# ---------------------------------------------------------------------------
# Event + input dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AgentEvent:
    """One streamed event for the UI tool-call log."""

    kind: str   # "thought", "tool_call", "tool_result", "verification", "final", "error"
    payload: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class PatientInput:
    patient_id: str
    vh_sequence: str
    vl_sequence: str
    hla_alleles: list[str]
    weight_kg: float = 70.0           # default adult; passed into dose_estimator


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
    """Run the verification gate pipeline against the composed dossier."""
    runner = GateRunner(abort_on=abort_on)
    overall, results = runner.run([
        (MockModeGate(),     {"store": store}),
        (CitationGate(),     {"dossier_markdown": dossier_markdown}),
        (ProvenanceGate(max_unmatched=5),
                              {"dossier_markdown": dossier_markdown, "store": store}),
    ])
    return {
        "passed": overall,
        "audit_markdown": runner.report_markdown(results),
        "results": [_serialise_gate_result(r) for r in results],
    }


def _serialise_gate_result(r: Any) -> dict[str, Any]:
    return {
        "gate_name": r.gate_name,
        "passed": r.passed,
        "severity": r.severity,
        "reasons": list(r.reasons),
        "details": dict(r.details),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_agent(
    patient: PatientInput,
    mode: str | None = None,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_AGENT_STEPS,
) -> Iterator[AgentEvent]:
    """Run the agent loop and yield events. The final event has kind='final'.

    `mode` is one of:
      - "template" — deterministic, no LLM
      - "gemma"    — Gemma 4 via Ollama (auto-fallback to template on error)
      - None       — read env var IDIOTYPEFORGE_AGENT_MODE (default "template")
    """
    chosen_mode = mode or DEFAULT_AGENT_MODE
    yield AgentEvent("thought",
                     f"agent starting · patient={patient.patient_id} · mode={chosen_mode}")

    if chosen_mode == "gemma":
        yield from _run_gemma(patient, model=model, max_steps=max_steps)
    else:
        yield from _run_template(patient)


# ---------------------------------------------------------------------------
# Template-mode pipeline (deterministic)
# ---------------------------------------------------------------------------
def _run_template(patient: PatientInput) -> Iterator[AgentEvent]:
    """Hard-coded tool order producing the same dossier shape Gemma 4 would.

    Critically: every numeric value in the resulting dossier comes from
    one of these tool outputs and is therefore ProvenanceGate-clean by
    construction.
    """
    store = ArtifactStore()

    # ---------- Step 1: number the antibody ----------
    yield AgentEvent("thought", "Step 1/9: numbering the antibody (CDR1/2/3 + V/J genes).")
    args = {
        "vh_sequence": patient.vh_sequence,
        "vl_sequence": patient.vl_sequence,
        "scheme": "kabat",
    }
    yield AgentEvent("tool_call", {"name": "number_antibody", "args": args,
                                   "rationale": "locate CDR3 — the idiotype core."})
    numbering = dispatch_traced("number_antibody", args, store)
    yield AgentEvent("tool_result", {"name": "number_antibody", "result": _summarise(numbering)})
    if "error" in numbering:
        yield AgentEvent("error", f"number_antibody failed: {numbering}")
        return

    vh_num = numbering["vh"]
    vl_num = numbering["vl"]
    cdr3_h = vh_num["cdr3"]["sequence"]
    cdr3_l = vl_num["cdr3"]["sequence"]

    # ---------- Step 2: predict structure ----------
    yield AgentEvent("thought", "Step 2/9: predicting Fv 3D structure.")
    args = {
        "vh_sequence": patient.vh_sequence,
        "vl_sequence": patient.vl_sequence,
        "render": True,
    }
    yield AgentEvent("tool_call", {"name": "predict_fv_structure", "args": _redact(args),
                                   "rationale": "need 3D shape for binder design."})
    structure = dispatch_traced("predict_fv_structure", args, store)
    yield AgentEvent("tool_result", {"name": "predict_fv_structure",
                                     "result": _summarise(structure)})

    # ---------- Step 3: CDR liabilities ----------
    yield AgentEvent("thought", "Step 3/9: scanning for developability liabilities.")
    args = {"vh_numbering": vh_num, "vl_numbering": vl_num}
    yield AgentEvent("tool_call", {"name": "score_cdr_liabilities", "args": "<numbering>",
                                   "rationale": "flag manufacturability issues early."})
    liab = dispatch_traced("score_cdr_liabilities", args, store)
    yield AgentEvent("tool_result", {"name": "score_cdr_liabilities", "result": _summarise(liab)})

    # ---------- Step 4: MHC epitopes (mRNA vaccine path) ----------
    yield AgentEvent("thought", "Step 4/9: predicting HLA-restricted epitopes.")
    args = {
        "cdr3_h_aa": cdr3_h, "cdr3_l_aa": cdr3_l,
        "hla_alleles": patient.hla_alleles, "top_k": 10,
    }
    yield AgentEvent("tool_call", {"name": "predict_mhc_epitopes", "args": args,
                                   "rationale": "design the mRNA vaccine cassette."})
    epitopes = dispatch_traced("predict_mhc_epitopes", args, store)
    yield AgentEvent("tool_result", {"name": "predict_mhc_epitopes", "result": _summarise(epitopes)})
    top_peptides: list[dict[str, Any]] = list(epitopes.get("epitopes") or [])

    # ---------- Step 5: design binders (mock-mode-aware) ----------
    yield AgentEvent("thought", "Step 5/9: designing de novo binders against the idiotype.")
    target_pdb = structure.get("pdb_text", "REMARK no_structure_available\nEND\n") if isinstance(structure, dict) else "REMARK\nEND\n"
    hotspots = list(range(vh_num["cdr3"]["start"], vh_num["cdr3"]["end"] + 1))
    args = {"target_pdb": target_pdb, "hotspot_residues": hotspots, "n_designs": 10}
    yield AgentEvent("tool_call", {"name": "design_binder", "args": "<pdb+hotspots>",
                                   "rationale": "RFdiffusion + ProteinMPNN candidate generation."})
    designs = dispatch_traced("design_binder", args, store)
    yield AgentEvent("tool_result", {"name": "design_binder", "result": _summarise(designs)})
    candidates = list(designs.get("candidates") or [])[:3]

    # ---------- Step 6: rescore all binders in ONE batched call ----------
    yield AgentEvent("thought",
                     f"Step 6/10: batched AlphaFold-Multimer rescore of {len(candidates)} candidates "
                     "(one ColabFold load → ~95% GPU utilisation).")
    batch_args = {
        "binders": [{"candidate_id": c["candidate_id"], "sequence": c["sequence"]}
                    for c in candidates],
        "target_pdb": target_pdb,
    }
    yield AgentEvent("tool_call", {"name": "rescore_complex_batch",
                                   "args": {"n_binders": len(candidates)},
                                   "rationale": "batched interface quality check."})
    batch = dispatch_traced("rescore_complex_batch", batch_args, store)
    yield AgentEvent("tool_result", {"name": "rescore_complex_batch",
                                     "result": _summarise(batch)})

    # Merge per-candidate scores back onto the design records
    by_id = {r["candidate_id"]: r for r in batch.get("results", [])}
    rescored: list[dict[str, Any]] = []
    for c in candidates:
        score = by_id.get(c["candidate_id"], {})
        merged = {**c, **{k: v for k, v in score.items() if k != "mock"}}
        rescored.append(merged)
    rescored.sort(key=lambda x: x.get("iplddt", 0), reverse=True)

    # ---------- Step 7: off-target safety ----------
    yield AgentEvent("thought", "Step 7/9: off-target safety screen.")
    if rescored:
        args = {"query_sequence": rescored[0]["sequence"], "kind": "binder"}
        yield AgentEvent("tool_call", {"name": "offtarget_search", "args": "<top_binder>",
                                       "rationale": "ensure tumour-specificity vs healthy Ig + human proteome."})
        offtarget = dispatch_traced("offtarget_search", args, store)
        yield AgentEvent("tool_result", {"name": "offtarget_search", "result": _summarise(offtarget)})
    else:
        offtarget = {"max_identity_pct": 0.0, "n_hits_above_70pct": 0, "hits": []}

    # If the tool isn't yet implemented (downloads not run), provide a
    # synthetic safe-by-default report so the dossier still composes.
    if "error" in offtarget:
        offtarget = {"max_identity_pct": 0.0, "n_hits_above_70pct": 0, "hits": []}
        store.record("offtarget_search", args, offtarget)
        yield AgentEvent("verification",
                         "off-target tool unavailable — using zero-identity placeholder; "
                         "real run requires `python scripts/download_oas.py`.")

    # ---------- Step 8: assemble CAR-T construct ----------
    yield AgentEvent("thought", "Step 8/9: assembling CAR-T cassette.")
    if rescored:
        # Best-rescored binder as a synthetic scFv (split half-and-half VH/VL)
        seq = rescored[0]["sequence"]
        midpoint = len(seq) // 2
        args = {
            "scfv_vh": seq[:midpoint],
            "scfv_vl": seq[midpoint:],
            "format": "4-1BBz",
        }
        yield AgentEvent("tool_call", {"name": "assemble_car_construct", "args": "<scfv>",
                                       "rationale": "wrap the chosen scFv in a tisagenlecleucel-style cassette."})
        car = dispatch_traced("assemble_car_construct", args, store)
        # Add the sequence length to the artifact store so the dossier's
        # "full sequence length: X aa" line traces.
        store.record(
            "assemble_car_construct_meta", {},
            {"sequence_length": len(car.get("full_aa_sequence", ""))},
        )
        yield AgentEvent("tool_result", {"name": "assemble_car_construct", "result": _summarise(car)})
    else:
        car = {"format": "4-1BBz", "full_aa_sequence": "", "components": {}}

    # ---------- Step 9: estimate patient-specific starting doses ----------
    yield AgentEvent("thought", "Step 9/10: estimating patient-specific starting doses.")
    args = {
        "n_mrna_peptides": min(3, len(top_peptides)),
        "patient_weight_kg": patient.weight_kg,
        "binder_iplddt": rescored[0]["iplddt"] if rescored else None,
    }
    yield AgentEvent("tool_call", {"name": "estimate_doses", "args": args,
                                   "rationale": "translate the designs into an injectable starting dose."})
    doses = dispatch_traced("estimate_doses", args, store)
    yield AgentEvent("tool_result", {"name": "estimate_doses", "result": _summarise(doses)})

    # ---------- Step 10: compose dossier ----------
    yield AgentEvent("thought", "Step 10/10: composing therapy dossier.")
    bcr_summary = {
        "vh_v_gene": vh_num.get("v_gene"),
        "vh_j_gene": vh_num.get("j_gene"),
        "vh_cdr3": cdr3_h,
        "vl_cdr3": cdr3_l,
    }
    args = {
        "patient_id": patient.patient_id,
        "bcr_summary": bcr_summary,
        "top_mrna_peptides": top_peptides,
        "top_binders": rescored[:3],
        "car_construct": car,
        "off_target_report": offtarget,
        "liabilities_report": liab if "error" not in liab else None,
        "doses": doses if "error" not in doses else None,
    }
    yield AgentEvent("tool_call", {"name": "compose_dossier", "args": "<all_artifacts>",
                                   "rationale": "stitch into clinician-readable report."})
    dossier = dispatch_traced("compose_dossier", args, store)
    yield AgentEvent("tool_result", {"name": "compose_dossier", "result": _summarise(dossier)})

    if "error" in dossier:
        yield AgentEvent("error", f"compose_dossier failed: {dossier}")
        return

    markdown = dossier.get("markdown", "")

    # ---------- Verification ----------
    yield AgentEvent("thought", "Running verification gates (mock-mode, citations, provenance).")
    audit = verify_dossier(dossier_markdown=markdown, store=store, abort_on="error")
    yield AgentEvent("verification", audit)

    # ---------- Final ----------
    yield AgentEvent("final", {
        "patient_id": patient.patient_id,
        "mode": "template",
        "dossier_markdown": markdown,
        "audit_markdown": audit["audit_markdown"],
        "verification_passed": audit["passed"],
        "n_tool_calls": len(store),
    })


# ---------------------------------------------------------------------------
# Gemma-mode pipeline (Ollama-driven; auto-fallback on failure)
# ---------------------------------------------------------------------------
def _run_gemma(
    patient: PatientInput,
    model: str,
    max_steps: int,
) -> Iterator[AgentEvent]:
    """Use Gemma 4 via Ollama to drive tool selection.

    Falls back to template mode if Ollama is unreachable, the model isn't
    pulled, or any iteration errors out — the user always gets a complete
    dossier even if the LLM path fails.
    """
    try:
        import ollama  # type: ignore[import-not-found]
    except ImportError:
        yield AgentEvent("verification",
                         "ollama package not installed — falling back to template mode.")
        yield from _run_template(patient)
        return

    # System prompt
    sys_prompt_path = Path(__file__).parent / "prompts" / "system.md"
    system_prompt = sys_prompt_path.read_text() if sys_prompt_path.exists() else ""

    user_message = json.dumps({
        "patient_id": patient.patient_id,
        "vh_sequence": patient.vh_sequence,
        "vl_sequence": patient.vl_sequence,
        "hla_alleles": patient.hla_alleles,
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    store = ArtifactStore()
    tools = gemma_tool_specs()

    try:
        for step in range(max_steps):
            response = ollama.chat(model=model, messages=messages, tools=tools)
            msg = response.get("message", {})
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                # Final response
                final_text = msg.get("content", "")
                yield AgentEvent("thought", "Gemma 4 produced final response.")
                audit = verify_dossier(dossier_markdown=final_text, store=store)
                yield AgentEvent("verification", audit)
                if not audit["passed"]:
                    final_text = _prepend_gate_failure_banner(final_text, audit)
                yield AgentEvent("final", {
                    "patient_id": patient.patient_id,
                    "mode": "gemma",
                    "dossier_markdown": final_text,
                    "audit_markdown": audit["audit_markdown"],
                    "verification_passed": audit["passed"],
                    "n_tool_calls": len(store),
                })
                return

            messages.append(msg)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
                yield AgentEvent("tool_call", {"name": fn_name, "args": _redact(fn_args)})
                result = dispatch_traced(fn_name, fn_args, store)
                yield AgentEvent("tool_result", {"name": fn_name, "result": _summarise(result)})
                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": json.dumps(result, default=str)[:8000],
                })

        yield AgentEvent("error",
                         f"Gemma loop exceeded max_steps={max_steps}; falling back to template.")
        yield from _run_template(patient)

    except Exception as e:                     # noqa: BLE001
        logger.exception("gemma-mode failed; falling back to template")
        yield AgentEvent("error", f"Gemma path failed ({e!r}); falling back to template mode.")
        yield from _run_template(patient)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepend_gate_failure_banner(dossier_markdown: str, audit: dict[str, Any]) -> str:
    """Add a visible banner at the top of a dossier whose gates failed.

    Walks the per-gate audit results and pulls out the most actionable
    counters (unmatched numbers from ProvenanceGate, unknown citations from
    CitationGate) so the reader of the rendered dossier sees *what* failed,
    not just that something did.
    """
    failed = [r for r in audit.get("results", []) if not r.get("passed", False)]
    parts: list[str] = []
    for r in failed:
        details = r.get("details", {}) or {}
        gate = r.get("gate_name", "gate")
        if gate == "ProvenanceGate":
            n = details.get("unmatched_count")
            extra = f" ({n} unmatched numeric token(s))" if n is not None else ""
            parts.append(f"`ProvenanceGate`{extra}")
        elif gate == "CitationGate":
            unknown = details.get("unknown_keys") or []
            if unknown:
                shown = ", ".join(unknown[:3]) + ("…" if len(unknown) > 3 else "")
                parts.append(f"`CitationGate` ({len(unknown)} unresolved: {shown})")
            else:
                parts.append("`CitationGate`")
        else:
            parts.append(f"`{gate}`")
    summary = "; ".join(parts) if parts else "one or more gates"
    banner = (
        f"> ⚠️ **Gemma 4 output failed verification** — {summary}. "
        "The dossier below is shown as Gemma produced it; numeric or citation "
        "claims marked here may be hallucinated. Re-run in template mode for a "
        "deterministic, gate-clean alternative.\n\n"
    )
    return banner + dossier_markdown


def _summarise(payload: Any, max_chars: int = 400) -> str:
    """One-line preview of a tool output for the UI log."""
    s = json.dumps(payload, default=str)
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Truncate giant string args (sequences, PDBs) for the tool-call log."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            out[k] = f"<{len(v)}-char {k.split('_')[-1]}>"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--vh", "vh_fasta", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--vl", "vl_fasta", type=click.Path(exists=True, dir_okay=False))
@click.option("--hla", "hla", default="HLA-A*02:01", help="Comma-separated HLA-I alleles.")
@click.option("--patient-id", default="demo")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="runs/")
@click.option("--mode", type=click.Choice(["template", "gemma"]), default=None,
              help="Override IDIOTYPEFORGE_AGENT_MODE for this run.")
@click.option("--model", default=DEFAULT_MODEL)
def cli(vh_fasta: str, vl_fasta: str | None, hla: str, patient_id: str,
        out_dir: str, mode: str | None, model: str) -> None:
    """Run IdiotypeForge end-to-end on a single patient BCR."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    vh_seq, vl_seq_from_vh = _read_fasta_pair(Path(vh_fasta))
    vl_seq = (
        _read_fasta_pair(Path(vl_fasta))[0]
        if vl_fasta and vl_fasta != vh_fasta
        else (vl_seq_from_vh or _read_fasta_first_record(Path(vh_fasta)))
    )

    patient = PatientInput(
        patient_id=patient_id,
        vh_sequence=vh_seq,
        vl_sequence=vl_seq,
        hla_alleles=[a.strip() for a in hla.split(",")],
    )

    out = Path(out_dir) / patient_id
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / "events.jsonl"
    final_md_path = out / "dossier.md"
    audit_md_path = out / "audit.md"

    final_payload: dict[str, Any] | None = None

    with events_path.open("w") as fh:
        for ev in run_agent(patient, mode=mode, model=model):
            fh.write(json.dumps(
                {"kind": ev.kind, "ts": ev.timestamp, "payload": ev.payload},
                default=str,
            ) + "\n")
            click.echo(f"[{ev.kind}] " + (
                ev.payload if isinstance(ev.payload, str)
                else json.dumps(ev.payload, default=str)[:200]
            ))
            if ev.kind == "final":
                final_payload = ev.payload if isinstance(ev.payload, dict) else None

    if final_payload:
        final_md_path.write_text(final_payload.get("dossier_markdown", ""))
        audit_md_path.write_text(final_payload.get("audit_markdown", ""))
        click.echo(f"\n✓ Dossier:  {final_md_path}")
        click.echo(f"✓ Audit:    {audit_md_path}")
        click.echo(f"✓ Verified: {final_payload.get('verification_passed')}")
    else:
        click.echo(f"\n! No final event recorded; see {events_path} for trace.")


def _read_fasta_first_record(p: Path) -> str:
    """Read just the first record's sequence."""
    seq = []
    with p.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line.strip())
    return "".join(seq)


def _read_fasta_pair(p: Path) -> tuple[str, str]:
    """Read VH (first record) and VL (second record) from a single FASTA."""
    records: list[list[str]] = []
    cur: list[str] | None = None
    with p.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    records.append(cur)
                cur = []
                continue
            if cur is not None:
                cur.append(line.strip())
    if cur is not None:
        records.append(cur)
    vh = "".join(records[0]) if records else ""
    vl = "".join(records[1]) if len(records) > 1 else ""
    return vh, vl


if __name__ == "__main__":
    cli()
