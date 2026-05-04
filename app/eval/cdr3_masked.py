"""CDR3-masked-AA top-1 accuracy benchmark.

Measures how well a (base or fine-tuned) Gemma 4 model predicts the
amino-acid identity at masked CDR3-H positions.

Plan claim:
    base Gemma 4 E4B  → ~25% top-1 (approx; baseline assumption)
    OAS-tuned LoRA    → ≥50% top-1

Heavy-only mode: OAS records persist heavy-chain sequences. ANARCI requires
a paired (VH, VL) input; we feed a benign κ-framework placeholder VL so the
heavy-chain numbering still works without inventing light-chain data.

This module is designed to import on CPU for unit testing — torch is
imported lazily and only the benchmark loop touches GPU primitives.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol


# A short, conserved κ-light framework used purely as an ANARCI partner.
# It is NOT used for any prediction — it lets `number_antibody.run` accept
# a heavy-only input without us patching its signature.
_KAPPA_FRAMEWORK_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQGISSALAWYQQKPGKAPKLLIYDASSLES"
    "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQFNSYPLTFGGGTKVEIK"
)

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_ANTIBODY_OPEN = "<antibody>\n"


class _GenerativeModel(Protocol):
    """Minimal protocol the benchmark needs from the model under test."""

    def generate(self, **kwargs: Any) -> Any: ...


def _raw_sequence(record: dict[str, Any]) -> str | None:
    """Pull the raw amino-acid sequence from a fine-tune dataset record.

    Records may carry a 'raw' field (preferred) or only the wrapped
    "<antibody>\nSEQ\n</antibody>" text field; we tolerate both.
    """
    if "raw" in record and isinstance(record["raw"], str):
        return record["raw"].strip()
    text = record.get("text")
    if not isinstance(text, str):
        return None
    if _ANTIBODY_OPEN in text:
        body = text.split(_ANTIBODY_OPEN, 1)[1]
        body = body.split("</antibody>", 1)[0]
        return body.strip()
    return text.strip()


def _cdr3_h_span(seq: str) -> tuple[int, int] | None:
    """Return (start_0based, end_0based_exclusive) for CDR3-H, or None on failure."""
    from app.tools.number_antibody import run as anarci_run

    try:
        out = anarci_run(
            vh_sequence=seq,
            vl_sequence=_KAPPA_FRAMEWORK_VL,
            scheme="kabat",
        )
    except Exception:                                   # noqa: BLE001
        return None
    cdr3 = (out.get("vh") or {}).get("cdr3") or {}
    start = cdr3.get("start")
    end = cdr3.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    # ANARCI emits 1-based inclusive — convert to 0-based half-open.
    s = max(0, start - 1)
    e = min(len(seq), end)
    if s >= e:
        return None
    return s, e


def _greedy_next_aa(
    model: _GenerativeModel, tokenizer: Any, prefix: str
) -> str | None:
    """Greedy-decode one token at ``prefix`` and return its first AA character.

    Returns None if the decoded text contains no AA character (e.g. the model
    emitted a punctuation token or a multi-byte symbol).
    """
    import torch                                        # type: ignore[import-not-found]

    inputs = tokenizer(prefix, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", 0),
        )
    new_token_ids = out[0, input_len:]
    if len(new_token_ids) == 0:
        return None
    decoded = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    for ch in decoded:
        if ch in _AA_ALPHABET:
            return ch
    return None


def compute_cdr3_masked_top1(
    model: Any,
    tokenizer: Any,
    eval_records: Iterable[dict[str, Any]],
    n_seqs: int = 500,
) -> dict[str, Any]:
    """For each (heavy-chain) record, mask one CDR3-H residue at a time and
    score whether the model greedy-decodes the right amino acid.

    Returns a metrics dict:
        {
            "n_sequences_attempted": int,
            "n_sequences_scored":    int,    # excluding ANARCI failures
            "n_positions":           int,    # total masked positions
            "n_correct":             int,
            "top1_accuracy":         float,  # n_correct / n_positions
            "n_anarci_failures":     int,
        }
    """
    n_attempted = 0
    n_scored = 0
    n_anarci_fail = 0
    total_positions = 0
    total_correct = 0

    for rec in eval_records:
        if n_attempted >= n_seqs:
            break
        n_attempted += 1
        seq = _raw_sequence(rec)
        if not seq or len(seq) < 50:
            continue
        span = _cdr3_h_span(seq)
        if span is None:
            n_anarci_fail += 1
            continue
        start, end = span
        n_scored += 1
        for i in range(start, end):
            prefix = _ANTIBODY_OPEN + seq[:i]
            predicted = _greedy_next_aa(model, tokenizer, prefix)
            total_positions += 1
            if predicted is not None and predicted == seq[i]:
                total_correct += 1

    accuracy = total_correct / total_positions if total_positions else 0.0
    return {
        "n_sequences_attempted": n_attempted,
        "n_sequences_scored": n_scored,
        "n_positions": total_positions,
        "n_correct": total_correct,
        "top1_accuracy": accuracy,
        "n_anarci_failures": n_anarci_fail,
    }
