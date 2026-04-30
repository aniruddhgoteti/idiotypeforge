"""Concrete verification gates and the runner that orchestrates them.

Every gate is a pure function on (input_artefacts) → GateResult. The runner
collects all results and produces an audit report attached to the dossier.

The gates run in the following intended order:

    1. SchemaGate         — tool output validates against pydantic model
    2. MockModeGate       — no `mock=True` markers in production runs
    3. ThresholdGate      — plan §4 numerical thresholds
    4. ProvenanceGate     — every number in the dossier appears in a tool output
                            (catches Gemma 4 hallucinating values)
    5. CitationGate       — every citation key resolves to references.bib

A failed gate doesn't always abort: severity is one of
  "info" / "warning" / "error" / "critical"
and the runner enforces an `abort_on` threshold (default "error").
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.verification.provenance import ArtifactStore, numeric_aliases


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
Severity = str   # Literal["info", "warning", "error", "critical"]


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    severity: Severity = "info"
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate base class
# ---------------------------------------------------------------------------
class Gate(ABC):
    name: str = "Gate"

    @abstractmethod
    def check(self, **kwargs: Any) -> GateResult: ...


# ---------------------------------------------------------------------------
# 1. SchemaGate
# ---------------------------------------------------------------------------
class SchemaGate(Gate):
    """Validates a tool output against a pydantic model."""

    name = "SchemaGate"

    def __init__(self, model: type[BaseModel]) -> None:
        self.model = model

    def check(self, output: Any, **_: Any) -> GateResult:
        try:
            self.model.model_validate(output)
            return GateResult(self.name, passed=True)
        except ValidationError as e:
            return GateResult(
                self.name,
                passed=False,
                severity="error",
                reasons=[f"Output failed pydantic validation: {len(e.errors())} errors"],
                details={"errors": e.errors()[:5]},
            )


# ---------------------------------------------------------------------------
# 2. MockModeGate
# ---------------------------------------------------------------------------
class MockModeGate(Gate):
    """Detects mock outputs in production runs.

    Behaviour:
      - If env IDIOTYPEFORGE_USE_MOCKS=1 (default): mocks are expected; emit
        an "info" result naming all mocked tools.
      - If env IDIOTYPEFORGE_USE_MOCKS=0: any mocked output is a "critical"
        violation — production runs must use real tools.
    """

    name = "MockModeGate"

    def check(self, store: ArtifactStore, **_: Any) -> GateResult:
        in_prod = os.environ.get("IDIOTYPEFORGE_USE_MOCKS", "1") == "0"
        mocked = store.mock_artifacts()
        mock_ids = [a.artifact_id for a in mocked]

        if not mocked:
            return GateResult(self.name, passed=True, reasons=["No mocked outputs."])

        if in_prod:
            return GateResult(
                self.name,
                passed=False,
                severity="critical",
                reasons=[
                    f"Production run (mocks disabled) but {len(mocked)} mock outputs reached pipeline.",
                ],
                details={"mock_artifacts": mock_ids},
            )

        return GateResult(
            self.name,
            passed=True,
            severity="info",
            reasons=[f"{len(mocked)} mock(s) used in dev mode (expected)."],
            details={"mock_artifacts": mock_ids},
        )


# ---------------------------------------------------------------------------
# 3. ThresholdGate
# ---------------------------------------------------------------------------
@dataclass
class Threshold:
    """A single named numeric threshold."""

    name: str
    json_path: str            # dotted path into a tool output, e.g. "$.mean_plddt"
    op: str                   # ">=" / "<=" / ">" / "<"
    value: float
    severity: Severity = "error"


_DEFAULT_THRESHOLDS: list[Threshold] = [
    Threshold("igfold.mean_plddt",       "$.mean_plddt",       ">=", 0.80),
    Threshold("igfold.cdr3_mean_plddt",  "$.cdr3_mean_plddt",  ">=", 0.65, severity="warning"),
    Threshold("rescore.iplddt",          "$.iplddt",           ">=", 0.50),
    Threshold("rescore.iplddt_warn",     "$.iplddt",           ">=", 0.70, severity="warning"),
    Threshold("rescore.ipae",            "$.ipae",             "<=", 15.0),
    Threshold("rescore.interface_sasa",  "$.interface_sasa",   ">=", 600.0, severity="warning"),
    Threshold("offtarget.max_identity",  "$.max_identity_pct", "<",  70.0),
]


class ThresholdGate(Gate):
    """Checks numerical thresholds against a tool output."""

    name = "ThresholdGate"

    def __init__(self, thresholds: list[Threshold] | None = None) -> None:
        self.thresholds = thresholds if thresholds is not None else _DEFAULT_THRESHOLDS

    def _resolve(self, payload: Any, path: str) -> Any:
        # path "$.iplddt"  or "$.foo.bar[0]"
        if not path.startswith("$"):
            return None
        cur: Any = payload
        for part in re.findall(r"\.([A-Za-z_][\w]*)|\[(\d+)\]", path[1:]):
            key, idx = part
            if key:
                if not isinstance(cur, dict) or key not in cur:
                    return None
                cur = cur[key]
            else:
                if not isinstance(cur, (list, tuple)) or int(idx) >= len(cur):
                    return None
                cur = cur[int(idx)]
        return cur

    @staticmethod
    def _compare(value: float, op: str, threshold: float) -> bool:
        return {
            ">=": value >= threshold,
            "<=": value <= threshold,
            ">":  value >  threshold,
            "<":  value <  threshold,
        }[op]

    def check(self, output: Any, threshold_subset: list[str] | None = None, **_: Any) -> GateResult:
        thresholds = self.thresholds
        if threshold_subset is not None:
            thresholds = [t for t in thresholds if t.name in threshold_subset]

        failures: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for t in thresholds:
            value = self._resolve(output, t.json_path)
            if value is None or not isinstance(value, (int, float)):
                continue
            if self._compare(float(value), t.op, t.value):
                continue
            entry = {
                "name": t.name,
                "json_path": t.json_path,
                "op": t.op,
                "threshold": t.value,
                "actual": float(value),
                "severity": t.severity,
            }
            (warnings if t.severity == "warning" else failures).append(entry)

        if failures:
            return GateResult(
                self.name,
                passed=False,
                severity="error",
                reasons=[f"{len(failures)} hard threshold(s) failed."],
                details={"failures": failures, "warnings": warnings},
            )
        if warnings:
            return GateResult(
                self.name,
                passed=True,
                severity="warning",
                reasons=[f"{len(warnings)} soft threshold(s) breached."],
                details={"warnings": warnings},
            )
        return GateResult(self.name, passed=True, reasons=["All thresholds met."])


# ---------------------------------------------------------------------------
# 4. ProvenanceGate — anti-hallucination
# ---------------------------------------------------------------------------
# Match a numeric token: integers, decimals, optional sign, optional unit suffix
# we strip before lookup ("%", "Å", "nM", "kcal").
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])(?P<num>-?\d+(?:[.,]\d+)?)(?:\s*(?:%|Å|nM|µM|kcal|kDa|aa|Da|h|hrs?))?"
)

# Numbers we ignore as not requiring provenance (years, list indices, formulae).
_BENIGN_NUMERIC_PATTERNS = [
    re.compile(r"\b(19|20)\d{2}\b"),     # years
    re.compile(r"\b[12]?\d\b"),           # one- or two-digit integers ≤ 29 (counts, lengths)
    re.compile(r"^[01]$"),                # 0 / 1
]


class ProvenanceGate(Gate):
    """Every number in the dossier text must trace back to a tool output.

    This is the core anti-hallucination check: if Gemma 4 invents a number
    (e.g. "ipLDDT was 0.91" when no tool returned 0.91), the gate fails.

    Behaviour:
      - Tokenise the dossier markdown for numeric tokens
      - Skip benign numbers (years, list indices, integers ≤ 29)
      - For every remaining number, build its alias set and check whether
        the alias index of the ArtifactStore contains it
      - Numbers that don't appear anywhere are flagged

    Tunables:
      - `max_unmatched`: tolerate up to N unmatched numbers (default 0)
      - `ignore_patterns`: extra regex patterns to whitelist
    """

    name = "ProvenanceGate"

    def __init__(
        self,
        max_unmatched: int = 0,
        ignore_patterns: list[re.Pattern[str]] | None = None,
    ) -> None:
        self.max_unmatched = max_unmatched
        self.ignore_patterns = list(_BENIGN_NUMERIC_PATTERNS) + (ignore_patterns or [])

    def _is_benign(self, raw: str) -> bool:
        # Use fullmatch only — a `search()` would match any digit substring
        # inside a longer numeric token (e.g. "0" inside "0.999"), spuriously
        # marking the whole token benign.
        return any(p.fullmatch(raw) for p in self.ignore_patterns)

    def check(self, dossier_markdown: str, store: ArtifactStore, **_: Any) -> GateResult:
        unmatched: list[dict[str, Any]] = []
        matched: list[dict[str, Any]] = []

        for m in _NUMBER_TOKEN.finditer(dossier_markdown):
            raw = m.group("num").replace(",", "")
            if self._is_benign(raw):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue

            aliases = numeric_aliases(value)
            hit = next(
                (
                    (a, store.lookup(a)[0])
                    for a in aliases
                    if store.has_alias(a)
                ),
                None,
            )
            if hit is None:
                unmatched.append({"value": value, "context": _context(dossier_markdown, m.start())})
            else:
                alias, (artifact_id, val, path) = hit
                matched.append({"value": value, "artifact_id": artifact_id, "json_path": path})

        if len(unmatched) > self.max_unmatched:
            return GateResult(
                self.name,
                passed=False,
                severity="critical",
                reasons=[
                    f"{len(unmatched)} numeric value(s) in dossier not traceable to any tool output."
                ],
                details={"unmatched": unmatched[:10], "matched_count": len(matched)},
            )
        return GateResult(
            self.name,
            passed=True,
            reasons=[
                f"All {len(matched)} dossier numbers trace to tool outputs "
                f"(tolerated unmatched: {len(unmatched)}/{self.max_unmatched})."
            ],
            details={"matched_count": len(matched), "unmatched_count": len(unmatched)},
        )


def _context(text: str, pos: int, span: int = 40) -> str:
    lo = max(0, pos - span)
    hi = min(len(text), pos + span)
    return text[lo:hi].replace("\n", " ")


# ---------------------------------------------------------------------------
# 5. CitationGate
# ---------------------------------------------------------------------------
_CITATION_TOKEN = re.compile(r"\[([A-Z][A-Za-z]+\d{4}[a-z]?(?:,\s*[A-Z][A-Za-z]+\d{4}[a-z]?)*)\]")


class CitationGate(Gate):
    """Every [Author Year] citation must resolve to a key in references.bib."""

    name = "CitationGate"

    def __init__(self, bib_path: str | Path = "data/references.bib") -> None:
        self.bib_path = Path(bib_path)

    def _load_keys(self) -> set[str]:
        if not self.bib_path.exists():
            return set()
        keys = set()
        for line in self.bib_path.read_text().splitlines():
            m = re.match(r"@\w+\s*\{\s*([^,\s]+)\s*,", line)
            if m:
                keys.add(m.group(1))
        return keys

    def check(self, dossier_markdown: str, **_: Any) -> GateResult:
        bib_keys = self._load_keys()
        if not bib_keys:
            return GateResult(
                self.name,
                passed=False,
                severity="error",
                reasons=[f"references.bib not found or empty at {self.bib_path}"],
            )

        unknown: list[str] = []
        seen: list[str] = []
        for m in _CITATION_TOKEN.finditer(dossier_markdown):
            for key in (k.strip() for k in m.group(1).split(",")):
                seen.append(key)
                if key not in bib_keys:
                    unknown.append(key)

        if unknown:
            return GateResult(
                self.name,
                passed=False,
                severity="error",
                reasons=[f"{len(unknown)} citation(s) do not resolve to references.bib."],
                details={
                    "unknown_keys": sorted(set(unknown)),
                    "available_keys_sample": sorted(bib_keys)[:10],
                },
            )
        return GateResult(
            self.name,
            passed=True,
            reasons=[f"All {len(seen)} citation(s) resolve."],
            details={"citations_found": sorted(set(seen))},
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class GateRunner:
    """Runs gates in order and produces an audit report.

    `abort_on` controls how strict the runner is about continuing past failures.
    Default: continue past warnings, abort on the first error/critical so the
    audit log shows exactly where the chain broke.
    """

    abort_on: Severity = "error"

    _SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "critical": 3}

    def run(
        self,
        gate_specs: list[tuple[Gate, dict[str, Any]]],
    ) -> tuple[bool, list[GateResult]]:
        """Each spec is (gate, kwargs_for_check).

        Returns (overall_passed, [results_in_order]).
        """
        threshold = self._SEVERITY_RANK[self.abort_on]
        results: list[GateResult] = []
        overall = True
        for gate, kwargs in gate_specs:
            r = gate.check(**kwargs)
            results.append(r)
            if not r.passed:
                overall = False
            if (not r.passed) and self._SEVERITY_RANK[r.severity] >= threshold:
                # stop early; remaining gates likely cascade
                break
        return overall, results

    def report_markdown(self, results: list[GateResult]) -> str:
        lines = ["## Verification audit", ""]
        for r in results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"- {icon} **{r.gate_name}** *(severity: {r.severity})*")
            for reason in r.reasons:
                lines.append(f"    - {reason}")
            if r.details:
                lines.append(f"    - details keys: `{list(r.details.keys())}`")
        return "\n".join(lines)
