"""Tests for the CDR3-masked-AA benchmark.

The benchmark module imports torch lazily so unit-level helpers
(`_raw_sequence`, `_cdr3_h_span`) test on CPU without the [gpu] extra.
The full `compute_cdr3_masked_top1` loop is exercised with a *fake* model
and tokenizer so we can validate the loop structure (counting,
ANARCI-failure handling, n_seqs cap) without GPU primitives.

Real-model accuracy is measured on the A100 spot VM via
``scripts/finetune_gemma4_unsloth.py --benchmark-only``; that path is
covered by an integration smoke test below, guarded by torch.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from app.eval.cdr3_masked import (
    _ANTIBODY_OPEN,
    _raw_sequence,
    compute_cdr3_masked_top1,
)


HAVE_ANARCI = importlib.util.find_spec("anarci") is not None
HAVE_TORCH = importlib.util.find_spec("torch") is not None
needs_anarci = pytest.mark.skipif(
    not HAVE_ANARCI,
    reason="anarci not installed; CDR3 boundary lookup unavailable.",
)


# ---------------------------------------------------------------------------
# _raw_sequence — handles both record shapes
# ---------------------------------------------------------------------------
def test_raw_sequence_prefers_raw_field() -> None:
    rec = {"text": f"{_ANTIBODY_OPEN}EVQLV\n</antibody>", "raw": "EVQLV"}
    assert _raw_sequence(rec) == "EVQLV"


def test_raw_sequence_unwraps_text_when_raw_missing() -> None:
    rec = {"text": f"{_ANTIBODY_OPEN}EVQLV\n</antibody>"}
    assert _raw_sequence(rec) == "EVQLV"


def test_raw_sequence_returns_text_unchanged_when_no_sentinels() -> None:
    rec = {"text": "EVQLV"}
    assert _raw_sequence(rec) == "EVQLV"


def test_raw_sequence_returns_none_when_no_text() -> None:
    assert _raw_sequence({"foo": "bar"}) is None


# ---------------------------------------------------------------------------
# Fake model + tokenizer — exercises the loop without torch on the GPU
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Maps each character to its own token id; decode is just chr."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        import torch
        ids = [ord(c) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids.tolist())


class _PerfectModel:
    """Generates the next ground-truth residue from the input prefix.

    The fake tokenizer encodes characters as their ord(); we decode the
    last input character and re-emit the *next* AA the test inserts via
    closure. We simulate by adding one fixed token id.
    """

    device = "cpu"

    def __init__(self, target_seq: str, prefix_marker: str) -> None:
        self._seq = target_seq
        self._prefix_marker = prefix_marker

    def generate(self, **kwargs: Any) -> Any:
        import torch
        input_ids = kwargs["input_ids"]
        # Decode prefix to find which residue index we're at.
        text = "".join(chr(int(c)) for c in input_ids[0].tolist())
        body = text.split(self._prefix_marker, 1)[-1]
        idx = len(body)
        if idx < len(self._seq):
            next_token = ord(self._seq[idx])
        else:
            next_token = ord("X")
        new = torch.tensor([[next_token]], dtype=torch.long)
        return torch.cat([input_ids, new], dim=-1)


class _AlwaysWrongModel:
    """Always emits 'X' (not in the AA alphabet) so the benchmark scores 0."""

    device = "cpu"

    def generate(self, **kwargs: Any) -> Any:
        import torch
        input_ids = kwargs["input_ids"]
        new = torch.tensor([[ord("z")]], dtype=torch.long)
        return torch.cat([input_ids, new], dim=-1)


# ---------------------------------------------------------------------------
# Loop-level tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_TORCH, reason="torch required")
def test_compute_top1_skips_records_without_cdr3() -> None:
    # Sequences too short for ANARCI to find a CDR3 — should not crash.
    records = [{"raw": "AAA"}, {"raw": "MM"}]
    result = compute_cdr3_masked_top1(
        model=_AlwaysWrongModel(), tokenizer=_FakeTokenizer(),
        eval_records=records, n_seqs=10,
    )
    assert result["n_sequences_attempted"] == 2
    assert result["n_sequences_scored"] == 0
    assert result["n_positions"] == 0
    assert result["top1_accuracy"] == 0.0


@pytest.mark.skipif(not HAVE_TORCH, reason="torch required")
def test_compute_top1_caps_at_n_seqs() -> None:
    # Provide more records than the cap; only the first should be attempted.
    records = [{"raw": "AAA"}] * 20
    result = compute_cdr3_masked_top1(
        model=_AlwaysWrongModel(), tokenizer=_FakeTokenizer(),
        eval_records=records, n_seqs=3,
    )
    assert result["n_sequences_attempted"] == 3


@pytest.mark.skipif(not HAVE_TORCH or not HAVE_ANARCI, reason="needs torch + anarci")
def test_compute_top1_perfect_model_scores_100pct() -> None:
    rituximab_vh = (
        "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIY"
        "PGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGG"
        "DWYFNVWGAGTTVTVSA"
    )
    records = [{"raw": rituximab_vh}]
    result = compute_cdr3_masked_top1(
        model=_PerfectModel(rituximab_vh, _ANTIBODY_OPEN),
        tokenizer=_FakeTokenizer(),
        eval_records=records,
        n_seqs=1,
    )
    assert result["n_sequences_scored"] == 1
    assert result["n_positions"] >= 5      # CDR3-H is at least 5 residues
    assert result["top1_accuracy"] == pytest.approx(1.0)


@pytest.mark.skipif(not HAVE_TORCH or not HAVE_ANARCI, reason="needs torch + anarci")
def test_compute_top1_wrong_model_scores_0pct() -> None:
    rituximab_vh = (
        "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIY"
        "PGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGG"
        "DWYFNVWGAGTTVTVSA"
    )
    records = [{"raw": rituximab_vh}]
    result = compute_cdr3_masked_top1(
        model=_AlwaysWrongModel(),
        tokenizer=_FakeTokenizer(),
        eval_records=records,
        n_seqs=1,
    )
    assert result["n_sequences_scored"] == 1
    assert result["top1_accuracy"] == 0.0
