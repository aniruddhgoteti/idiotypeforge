"""Tool registry for Gemma 4 native function calling.

Imports all 9 tool modules, collects their SCHEMA constants, and exposes a
single dispatch function the orchestrator can hand to Gemma 4.

Schema format follows the Ollama / OpenAI-compatible function-calling spec:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

Gemma 4 then emits tool_calls in its response, which the orchestrator routes
back to `dispatch(name, args)`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from app.tools import (
    car_assembler,
    cdr_liabilities,
    compose_dossier,
    igfold_predict,
    mhcflurry_predict,
    number_antibody,
    offtarget_search,
    render_structure,
    rescore_complex,
    rfdiffusion_design,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_TOOLS: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    "number_antibody":      (number_antibody.run,      number_antibody.SCHEMA),
    "predict_fv_structure": (igfold_predict.run,       igfold_predict.SCHEMA),
    "score_cdr_liabilities":(cdr_liabilities.run,      cdr_liabilities.SCHEMA),
    "predict_mhc_epitopes": (mhcflurry_predict.run,    mhcflurry_predict.SCHEMA),
    "design_binder":        (rfdiffusion_design.run,   rfdiffusion_design.SCHEMA),
    "rescore_complex":      (rescore_complex.run,      rescore_complex.SCHEMA),
    "offtarget_search":     (offtarget_search.run,     offtarget_search.SCHEMA),
    "assemble_car_construct":(car_assembler.run,       car_assembler.SCHEMA),
    "render_structure":     (render_structure.run,     render_structure.SCHEMA),
    "compose_dossier":      (compose_dossier.run,      compose_dossier.SCHEMA),
}


def gemma_tool_specs() -> list[dict[str, Any]]:
    """Return the tool list in Ollama / OpenAI-compatible function-calling format."""
    return [
        {"type": "function", "function": schema}
        for _, schema in _TOOLS.values()
    ]


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name with the supplied arguments.

    Wraps every tool call in:
      - a structured log line (for the streaming UI)
      - exception capture (returns `{"error": str(e)}` so the agent can recover)
    """
    if name not in _TOOLS:
        return {"error": f"unknown tool: {name}", "available": list(_TOOLS)}

    fn, _ = _TOOLS[name]
    logger.info("→ tool=%s args_keys=%s", name, list(args))
    try:
        result = fn(**args)
        logger.info("← tool=%s status=ok", name)
        return result
    except NotImplementedError as e:
        logger.warning("← tool=%s status=stub", name)
        return {"error": "stub_not_implemented", "detail": str(e)}
    except Exception as e:  # noqa: BLE001 — agent must see the error message
        logger.exception("← tool=%s status=error", name)
        return {"error": e.__class__.__name__, "detail": str(e)}


def list_tools() -> list[str]:
    return sorted(_TOOLS)
