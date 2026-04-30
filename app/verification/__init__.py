"""Verification gate framework for IdiotypeForge.

Every numeric claim in the final dossier must be traceable to a specific
tool-call's output. Every citation must resolve to references.bib. No mock
outputs may reach a "production" run. Thresholds from the plan §4 are
enforced at the candidate level.

This module is the difference between "AI research demo" and "auditable
scientific artifact". A clinician (or a sceptical judge) can re-run the
verification harness on the same artefact bundle and see exactly which
numbers came from which tool call, and whether anything was made up.

Public API:
    ArtifactStore           — accumulates tool inputs/outputs with stable IDs
    GateResult              — outcome record for a single gate
    Gate (abstract)         — gate interface
    SchemaGate              — pydantic conformance
    MockModeGate            — production safety
    ThresholdGate           — numeric thresholds
    ProvenanceGate          — anti-hallucination
    CitationGate            — every citation resolves
    GateRunner              — orchestrates gates in order
"""
from app.verification.gates import (
    CitationGate,
    Gate,
    GateResult,
    GateRunner,
    MockModeGate,
    ProvenanceGate,
    SchemaGate,
    ThresholdGate,
)
from app.verification.provenance import ArtifactStore

__all__ = [
    "ArtifactStore",
    "GateResult",
    "Gate",
    "SchemaGate",
    "MockModeGate",
    "ThresholdGate",
    "ProvenanceGate",
    "CitationGate",
    "GateRunner",
]
