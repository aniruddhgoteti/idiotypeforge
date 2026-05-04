"""Tests for the AbLang2 saliency helper.

The unit-level helpers (`attention_rollup`, `render_saliency_bar`) are
covered without optional deps; `extract_ablang2_attention` and the
end-to-end `compute_saliency_card` are guarded by importorskip on
`ablang2` so they slot into the existing 7-skipped set on CPU CI.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from app.ui.saliency import (
    attention_rollup,
    compute_saliency_card,
    render_saliency_bar,
)


HAVE_ABLANG2 = importlib.util.find_spec("ablang2") is not None
needs_ablang2 = pytest.mark.skipif(
    not HAVE_ABLANG2,
    reason="ablang2 not installed; run `uv sync --extra igfold`.",
)


# ---------------------------------------------------------------------------
# attention_rollup — pure numpy, no deps
# ---------------------------------------------------------------------------
def test_attention_rollup_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    attn = rng.random((4, 8, 20, 20))   # (L=4, H=8, S=20, S=20)
    cdr3_idx = [10, 11, 12, 13, 14]
    scores = attention_rollup(attn, cdr3_idx)
    assert scores.shape == (5,)
    assert scores.max() == pytest.approx(1.0)
    assert scores.min() >= 0.0


def test_attention_rollup_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        attention_rollup(np.zeros((4, 4, 4)), [0, 1])


def test_attention_rollup_handles_zero_input() -> None:
    attn = np.zeros((1, 1, 5, 5))
    scores = attention_rollup(attn, [1, 2, 3])
    assert scores.shape == (3,)
    # All zeros — the helper short-circuits the divide-by-zero branch.
    assert np.all(scores == 0.0)


# ---------------------------------------------------------------------------
# render_saliency_bar — base64 PNG, no deps
# ---------------------------------------------------------------------------
def test_render_saliency_bar_returns_png_b64() -> None:
    scores = np.array([0.1, 0.5, 1.0, 0.3])
    out = render_saliency_bar(scores, "ARNT")
    assert isinstance(out, str)
    assert len(out) > 100
    # base64 PNGs decoded start with the 8-byte PNG signature \x89PNG\r\n\x1a\n.
    import base64
    decoded = base64.b64decode(out)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# compute_saliency_card — graceful no-input + missing-dep paths
# ---------------------------------------------------------------------------
def test_compute_saliency_card_empty_input() -> None:
    out = compute_saliency_card("", "")
    assert "Paste a VH" in out


def test_compute_saliency_card_install_hint_when_no_ablang2() -> None:
    if HAVE_ABLANG2:
        pytest.skip("ablang2 is installed; this test only runs when it isn't.")
    out = compute_saliency_card("EVQLVQSGGGLVKPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGYYFDYWGQGTLVTVSS", "")
    assert "ablang2" in out.lower()


# ---------------------------------------------------------------------------
# Real AbLang2 forward pass — only runs when the optional dep is installed
# ---------------------------------------------------------------------------
@needs_ablang2
def test_extract_ablang2_attention_shapes() -> None:
    from app.ui.saliency import extract_ablang2_attention

    rituximab_vh = (
        "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA"
    )
    attn, residues, cdr3_idx = extract_ablang2_attention(rituximab_vh)
    assert attn.ndim == 4
    assert attn.shape[-1] == attn.shape[-2] == len(residues)
    assert all(0 <= i < len(residues) for i in cdr3_idx)
    assert len(cdr3_idx) >= 5      # CDR3-H is at least 5 residues
