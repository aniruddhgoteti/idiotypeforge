"""AbLang2 attention-based saliency over CDR3.

Rolls up the per-head self-attention from the AbLang2 antibody language
model across the CDR3-H region into a single per-residue importance vector,
then renders it as a horizontal bar chart and an HTML highlight string.

Used in the "Saliency" tab of the Gradio UI and in the dossier rationales.

Heavy deps (`ablang2`, `torch`) are imported lazily so this module loads
on CPU-only laptops without the optional `[igfold]` extra. The Gradio tab
shows an install-hint message when the deps are missing.
"""
from __future__ import annotations

import io
from base64 import b64encode

import matplotlib.pyplot as plt
import numpy as np


_INSTALL_HINT_MD = (
    "_Saliency requires the optional `ablang2` dependency._\n\n"
    "Install with: `uv sync --extra igfold` (also pulls IgFold weights).\n"
)


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


# ---------------------------------------------------------------------------
# AbLang2 attention extraction (lazy heavy deps)
# ---------------------------------------------------------------------------
def extract_ablang2_attention(vh: str) -> tuple[np.ndarray, str, list[int]]:
    """Run AbLang2 on the heavy chain and return attention + CDR3 indices.

    Returns
    -------
    attention_tensor : np.ndarray, shape (L, H, S, S)
        Stacked per-layer per-head self-attention over the residue tokens
        (special tokens stripped).
    residues : str
        The heavy-chain sequence aligned to the attention axes (length S).
    cdr3_indices : list[int]
        0-based indices into ``residues`` for the CDR3-H residues, derived
        from ANARCI Kabat numbering.
    """
    import torch                            # type: ignore[import-not-found]
    import ablang2                          # type: ignore[import-not-found]

    from app.tools.number_antibody import run as anarci_run

    seq = vh.strip().upper()
    if not seq:
        raise ValueError("Empty heavy-chain sequence.")

    # AbLang2's public API exposes a paired-chain model; we feed only the
    # heavy chain since CDR3-H carries the dominant idiotype signal.
    model = ablang2.pretrained("ablang2-paired")
    model.freeze()
    underlying = getattr(model, "AbLang", model)        # raw HF transformer
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("AbLang2 model is missing a `tokenizer` attribute.")

    encoded = tokenizer([seq], pad=True, w_extra_tkns=False, device="cpu")
    if isinstance(encoded, np.ndarray):
        encoded = torch.from_numpy(encoded)
    encoded = encoded.long()

    underlying.eval()
    with torch.no_grad():
        out = underlying(encoded, output_attentions=True, return_dict=True)
    attentions = out.attentions if hasattr(out, "attentions") else out["attentions"]
    # Each layer is (batch=1, heads, S, S). Stack → (L, heads, S, S).
    stacked = torch.stack([a[0] for a in attentions], dim=0).cpu().numpy()

    # Strip special tokens. AbLang2's tokeniser puts <|startoftext|> at index 0
    # and a separator/pad afterwards; we trim to len(seq) residue positions
    # starting at offset 1.
    seq_len = len(seq)
    # The encoded shape is (1, T); residue tokens are at [1 : 1 + seq_len].
    res_lo, res_hi = 1, 1 + seq_len
    if stacked.shape[-1] < res_hi:
        # Tokeniser produced fewer tokens than expected — fall back to a
        # contiguous trim of the full attention matrix.
        res_lo, res_hi = 0, stacked.shape[-1]
    attn_residues = stacked[:, :, res_lo:res_hi, res_lo:res_hi]

    # ANARCI for CDR3 boundaries. The public `run` requires both chains; we
    # pass a benign light-chain placeholder copied from a known κ framework
    # so the heavy-chain numbering still works.
    light_placeholder = (
        "DIQMTQSPSSLSASVGDRVTITCRASQGISSALAWYQQKPGKAPKLLIYDASSLESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQFNSYPLTFGGGTKVEIK"
    )
    numbering = anarci_run(vh_sequence=seq, vl_sequence=light_placeholder, scheme="kabat")
    cdr3 = numbering["vh"]["cdr3"]
    # 1-based gap-free → 0-based indices
    cdr3_indices = list(range(cdr3["start"] - 1, cdr3["end"]))
    return attn_residues, seq, cdr3_indices


def compute_saliency_card(vh: str, vl: str | None = None) -> str:
    """Top-level entry for the Gradio tab.

    Returns a markdown card. ``vl`` is accepted for symmetry with the input
    panel but ignored here: only CDR3-H is rendered. Empty input shows a
    placeholder; missing ``ablang2`` shows an install hint.
    """
    if not (vh or "").strip():
        return "_Paste a VH sequence (or load a demo case) and try again._\n"

    try:
        attention, residues, cdr3_idx = extract_ablang2_attention(vh)
    except ImportError:
        return _INSTALL_HINT_MD
    except Exception as e:                                  # noqa: BLE001
        return (
            f"_Saliency computation failed: `{type(e).__name__}: {str(e)[:200]}`._\n"
        )

    if not cdr3_idx:
        return "_ANARCI did not find a CDR3 in the heavy chain._\n"

    scores = attention_rollup(attention, cdr3_idx)
    cdr3_residues = "".join(residues[i] for i in cdr3_idx)
    png_b64 = render_saliency_bar(scores, cdr3_residues)
    return (
        "## CDR3-H attention saliency\n\n"
        "Per-residue attention received by each CDR3-H position, averaged "
        "across all AbLang2 layers and heads, normalised to [0, 1].\n\n"
        f"![saliency](data:image/png;base64,{png_b64})\n"
    )
