"""Headless structure rendering for Gemma 4's multimodal vision input.

CPU-only matplotlib 3D scatter plot of CA atoms, coloured by:
  - chain identity (H = blue, L = orange) for an Fv overview
  - pLDDT (red→yellow→green) when render `color_by_plddt=True`
  - hot-pink for residues passed in `highlight_residues`

Three view types are supported:
  - "overview"   — full structure
  - "cdr3"       — close-up around the highlighted residues
  - "interface"  — alias for overview when there's only one chain;
                    when there are two chains, picks contact zone

Returns base64-encoded PNGs ready to embed in Gemma 4 chat messages.

This is intentionally a "good-enough" renderer. The pretty PyMOL version
ships in the GPU phase if compute allows; the matplotlib renderer is the
always-on fallback.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt   # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402  (registers 3D projection)


SCHEMA = {
    "name": "render_structure",
    "description": (
        "Render headless PNG views of an antibody Fv or binder:antigen complex. "
        "Returns base64-encoded images suitable for Gemma 4 multimodal input."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pdb_text": {"type": "string"},
            "highlight_residues": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "1-based residue indices to highlight (e.g. CDR3).",
                "default": [],
            },
            "views": {
                "type": "array",
                "items": {"type": "string", "enum": ["overview", "cdr3", "interface"]},
                "default": ["overview", "cdr3"],
            },
            "color_by_plddt": {"type": "boolean", "default": True},
        },
        "required": ["pdb_text"],
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(
    pdb_text: str,
    highlight_residues: list[int] | None = None,
    views: list[str] | None = None,
    color_by_plddt: bool = True,
) -> dict[str, Any]:
    """Render the requested views and return {view_name: png_b64}."""
    highlights = set(highlight_residues or [])
    views_ = views or ["overview", "cdr3"]

    atoms = _parse_ca_atoms(pdb_text)
    if not atoms:
        raise ValueError(
            "No CA atoms parsed from pdb_text. Pass a valid PDB-formatted string."
        )

    out: dict[str, Any] = {"views": {}}
    for view in views_:
        if view == "overview":
            png = _render_overview(atoms, highlights, color_by_plddt)
        elif view == "cdr3":
            png = _render_close_up(atoms, highlights, color_by_plddt)
        elif view == "interface":
            png = _render_interface(atoms, highlights, color_by_plddt)
        else:
            continue
        out["views"][view] = png

    # Convenience flat fields so downstream tools can grab e.g. `result["overview"]`.
    out.update(out["views"])
    return out


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------
def _parse_ca_atoms(pdb_text: str) -> list[dict[str, Any]]:
    """Pull (chain, resseq, x, y, z, b_factor) for each CA. Tolerant of whitespace."""
    atoms: list[dict[str, Any]] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            chain = line[21:22].strip() or "A"
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            b = float(line[60:66]) if len(line) >= 66 else 0.0
        except (ValueError, IndexError):
            continue
        atoms.append({"chain": chain, "resseq": resseq, "x": x, "y": y, "z": z, "b": b})
    return atoms


# ---------------------------------------------------------------------------
# View renderers
# ---------------------------------------------------------------------------
def _render_overview(
    atoms: list[dict[str, Any]],
    highlights: set[int],
    color_by_plddt: bool,
) -> str:
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _draw_backbone(ax, atoms, highlights, color_by_plddt)
    ax.set_title("Fv overview" if _has_two_chains(atoms) else "Structure overview")
    return _fig_to_png_b64(fig)


def _render_close_up(
    atoms: list[dict[str, Any]],
    highlights: set[int],
    color_by_plddt: bool,
) -> str:
    """Crop to the highlighted residues + 6 Å buffer."""
    if highlights:
        focus_atoms = [a for a in atoms if a["resseq"] in highlights]
        if focus_atoms:
            cx = sum(a["x"] for a in focus_atoms) / len(focus_atoms)
            cy = sum(a["y"] for a in focus_atoms) / len(focus_atoms)
            cz = sum(a["z"] for a in focus_atoms) / len(focus_atoms)
            window = 12.0
            atoms = [a for a in atoms
                     if abs(a["x"] - cx) < window
                     and abs(a["y"] - cy) < window
                     and abs(a["z"] - cz) < window]
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _draw_backbone(ax, atoms, highlights, color_by_plddt, marker_size=80)
    ax.set_title("CDR3 close-up" if highlights else "Close-up")
    return _fig_to_png_b64(fig)


def _render_interface(
    atoms: list[dict[str, Any]],
    highlights: set[int],
    color_by_plddt: bool,
) -> str:
    """For two-chain structures, focus on the contact zone."""
    chains = sorted({a["chain"] for a in atoms})
    if len(chains) < 2:
        return _render_overview(atoms, highlights, color_by_plddt)

    chain_a, chain_b = chains[0], chains[1]
    a_atoms = [a for a in atoms if a["chain"] == chain_a]
    b_atoms = [a for a in atoms if a["chain"] == chain_b]
    contacts: set[int] = set()
    for aa in a_atoms:
        for bb in b_atoms:
            d2 = (aa["x"] - bb["x"]) ** 2 + (aa["y"] - bb["y"]) ** 2 + (aa["z"] - bb["z"]) ** 2
            if d2 < 8.0 ** 2:
                contacts.add(aa["resseq"])
                contacts.add(bb["resseq"])

    new_highlights = highlights | contacts
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _draw_backbone(ax, atoms, new_highlights, color_by_plddt)
    ax.set_title("Interface")
    return _fig_to_png_b64(fig)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
_CHAIN_COLORS = {"H": "#1f77b4", "L": "#ff7f0e", "K": "#ff7f0e", "A": "#1f77b4", "B": "#2ca02c"}


def _draw_backbone(
    ax: Any,
    atoms: list[dict[str, Any]],
    highlights: set[int],
    color_by_plddt: bool,
    marker_size: int = 30,
) -> None:
    if not atoms:
        return

    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]

    if color_by_plddt and any(a["b"] > 0 for a in atoms):
        # B-factor often holds pLDDT × 100. Normalise to [0, 1] for a cmap.
        cs = [max(0.0, min(1.0, a["b"] / 100.0)) for a in atoms]
        cmap = plt.get_cmap("RdYlGn")
        ax.scatter(xs, ys, zs, c=cs, cmap=cmap, vmin=0, vmax=1, s=marker_size, depthshade=True)
    else:
        cs = [_CHAIN_COLORS.get(a["chain"], "#777777") for a in atoms]
        ax.scatter(xs, ys, zs, c=cs, s=marker_size, depthshade=True)

    # Highlights: bigger pink markers on top
    if highlights:
        hxs, hys, hzs = [], [], []
        for a in atoms:
            if a["resseq"] in highlights:
                hxs.append(a["x"]); hys.append(a["y"]); hzs.append(a["z"])
        if hxs:
            ax.scatter(hxs, hys, hzs, c="#e91e63", s=marker_size * 2.5, marker="o", edgecolors="black")

    # Connect successive CAs with a thin line per chain
    by_chain: dict[str, list[dict[str, Any]]] = {}
    for a in atoms:
        by_chain.setdefault(a["chain"], []).append(a)
    for chain, chain_atoms in by_chain.items():
        chain_atoms.sort(key=lambda x: x["resseq"])
        ax.plot(
            [a["x"] for a in chain_atoms],
            [a["y"] for a in chain_atoms],
            [a["z"] for a in chain_atoms],
            color="#999999",
            linewidth=0.7,
            alpha=0.6,
        )

    ax.set_xlabel("Å"); ax.set_ylabel("Å"); ax.set_zlabel("Å")
    ax.tick_params(labelsize=7)


def _has_two_chains(atoms: list[dict[str, Any]]) -> bool:
    return len({a["chain"] for a in atoms}) >= 2


def _fig_to_png_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
