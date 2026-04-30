"""AbLang2 attention-based saliency over CDR3.

Rolls up the per-head self-attention from the AbLang2 antibody language
model across the CDR3 region into a single per-residue importance vector,
then renders it as a horizontal bar chart and an HTML highlight string.

Used in the "advanced" tab of the Gradio UI and in the dossier rationales.
"""
from __future__ import annotations

import io
from base64 import b64encode

import matplotlib.pyplot as plt
import numpy as np


def attention_rollup(attention_tensor: np.ndarray, cdr3_indices: list[int]) -> np.ndarray:
    """Aggregate (layers × heads × seq × seq) → (cdr3_len,).

    Mean across layers and heads, then for each CDR3 residue sum the
    attention it receives from all positions. Normalised to [0, 1].
    """
    if attention_tensor.ndim != 4:
        raise ValueError("Expected attention_tensor of shape (L, H, S, S).")
    avg = attention_tensor.mean(axis=(0, 1))             # (S, S)
    received = avg.sum(axis=0)                            # (S,)
    cdr3_scores = received[cdr3_indices]
    if cdr3_scores.max() > 0:
        cdr3_scores = cdr3_scores / cdr3_scores.max()
    return cdr3_scores


def render_saliency_bar(scores: np.ndarray, residues: str) -> str:
    """Return a base64 PNG of a CDR3 saliency bar chart."""
    fig, ax = plt.subplots(figsize=(max(4, len(scores) * 0.4), 2.0))
    ax.bar(range(len(scores)), scores, color="#1f77b4")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(list(residues), fontsize=10)
    ax.set_ylabel("attn (normalised)")
    ax.set_ylim(0, 1.1)
    ax.set_title("CDR3 attention saliency")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return b64encode(buf.getvalue()).decode()
