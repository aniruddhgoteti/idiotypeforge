"""Artifact store with numeric provenance tracking.

Every tool call in a pipeline run is stored with:
  - a stable artifact ID (`tool_name:call_index`)
  - the input arguments (as a deterministic JSON hash)
  - the output payload
  - a flat list of (numeric_value, json_path) pairs extracted from the output

The ProvenanceGate uses this index to detect numbers in the dossier text
that *don't* appear anywhere in the tool outputs — a strong signal that
the LLM hallucinated a value.

We deliberately avoid float exact-equality. Provenance match requires
floats to round-trip to a small set of canonical formats (3 sig figs,
2 sig figs, 1 decimal place, integer) — matching the way Gemma 4 typically
re-formats numbers in prose.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Canonical numeric formatting (covers how an LLM is likely to re-render a value)
# ---------------------------------------------------------------------------
def numeric_aliases(value: float | int) -> set[str]:
    """All plausible string representations a model might emit for `value`.

    Carefully avoids over-aggressive rounding (e.g. `f"{0.847:.0f}" == "1"`)
    that would collide unrelated values. We emit a `:.0f` alias **only** for
    actually-integer values; for non-integers we emit `:.1f` and finer.
    """
    if isinstance(value, bool):       # bool is a subclass of int — ignore
        return set()
    aliases: set[str] = set()
    fv = float(value)
    is_int = fv.is_integer()

    if is_int:
        aliases.add(str(int(fv)))
        aliases.add(f"{int(fv):,}")        # 12345 → "12,345"
        aliases.add(f"{fv:.0f}")
        aliases.add(f"{fv:.1f}")
        aliases.add(f"{fv:.2f}")
    else:
        # Non-integer: skip :.0f to avoid 0.847 → "1" collisions.
        aliases.add(f"{fv:.1f}")
        aliases.add(f"{fv:.2f}")
        aliases.add(f"{fv:.3f}")
        aliases.add(f"{fv:.4f}")
        # Significant-figure forms (e.g. 0.873 → "0.87")
        if fv != 0:
            aliases.add(f"{fv:.2g}")
            aliases.add(f"{fv:.3g}")

    # Percentage variants for fractions in [0, 1]: 0.325 → "32.5" / "33"
    if 0 <= abs(fv) <= 1:
        pct = fv * 100
        if pct.is_integer():
            aliases.add(f"{pct:.0f}")
        else:
            aliases.add(f"{pct:.0f}")           # rounded percent
            aliases.add(f"{pct:.1f}")
            aliases.add(f"{pct:.2f}")

    # Strip trailing zeros to catch "0.870" vs "0.87" vs "0.87"
    pruned = {a.rstrip("0").rstrip(".") if "." in a else a for a in aliases}
    return aliases | pruned


def walk_numbers(payload: Any, path: str = "$") -> Iterator[tuple[float, str]]:
    """Recursively yield (numeric_value, json_path) from a JSON-shaped payload."""
    if isinstance(payload, bool):
        return
    if isinstance(payload, (int, float)):
        yield float(payload), path
        return
    if isinstance(payload, dict):
        for k, v in payload.items():
            yield from walk_numbers(v, f"{path}.{k}")
        return
    if isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            yield from walk_numbers(v, f"{path}[{i}]")
        return
    # strings, None, bytes — ignore


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------
@dataclass
class Artifact:
    artifact_id: str          # e.g. "predict_fv_structure:0"
    tool_name: str
    call_index: int
    input_hash: str           # SHA-256 of canonical-JSON args
    output: Any
    is_mock: bool = False
    numeric_index: dict[str, list[tuple[float, str]]] = field(default_factory=dict)
    """Lookup from canonical alias → list of (value, json_path)."""


@dataclass
class ArtifactStore:
    """Accumulates tool calls and exposes a fast numeric provenance lookup."""
    artifacts: list[Artifact] = field(default_factory=list)
    _alias_index: dict[str, list[tuple[str, float, str]]] = field(default_factory=dict)
    """Global lookup: alias_string → [(artifact_id, value, json_path), ...]."""

    def record(
        self,
        tool_name: str,
        args: dict[str, Any],
        output: Any,
    ) -> Artifact:
        call_index = sum(1 for a in self.artifacts if a.tool_name == tool_name)
        artifact_id = f"{tool_name}:{call_index}"

        canonical_args = json.dumps(args, sort_keys=True, default=str)
        input_hash = hashlib.sha256(canonical_args.encode()).hexdigest()[:16]

        is_mock = bool(isinstance(output, dict) and output.get("mock") is True)

        numeric_index: dict[str, list[tuple[float, str]]] = {}
        for value, path in walk_numbers(output):
            for alias in numeric_aliases(value):
                numeric_index.setdefault(alias, []).append((value, path))
                self._alias_index.setdefault(alias, []).append((artifact_id, value, path))

        art = Artifact(
            artifact_id=artifact_id,
            tool_name=tool_name,
            call_index=call_index,
            input_hash=input_hash,
            output=output,
            is_mock=is_mock,
            numeric_index=numeric_index,
        )
        self.artifacts.append(art)
        return art

    def lookup(self, alias: str) -> list[tuple[str, float, str]]:
        """Return [(artifact_id, value, json_path)] for any tool call whose
        output contained a number with this canonical alias."""
        return list(self._alias_index.get(alias, []))

    def has_alias(self, alias: str) -> bool:
        return alias in self._alias_index

    def mock_artifacts(self) -> list[Artifact]:
        return [a for a in self.artifacts if a.is_mock]

    def __len__(self) -> int:
        return len(self.artifacts)
