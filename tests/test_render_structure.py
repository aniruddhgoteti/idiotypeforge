"""Tests for the headless structure renderer (matplotlib 3D)."""
from __future__ import annotations

import base64

import pytest

from app.tools.render_structure import _parse_ca_atoms, run as render_structure


# A tiny synthetic two-chain PDB. Real PDB columns: 1-6 ATOM, 7-11 serial,
# 13-16 atom name, 17 alt, 18-20 resname, 22 chain, 23-26 resseq,
# 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 b-factor.
TINY_PDB = """\
ATOM      1  CA  ALA H   1      10.000  10.000  10.000  1.00 90.00
ATOM      2  CA  GLY H   2      11.000  10.000  10.000  1.00 88.00
ATOM      3  CA  PHE H   3      12.000  10.000  10.000  1.00 70.00
ATOM      4  CA  TRP H   4      13.000  10.000  10.000  1.00 60.00
ATOM      5  CA  ALA L   1      10.000  12.000  10.000  1.00 86.00
ATOM      6  CA  ASP L   2      11.000  12.000  10.000  1.00 84.00
ATOM      7  CA  GLU L   3      12.000  12.000  10.000  1.00 82.00
END
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_ca_atoms_extracts_chains_and_coordinates() -> None:
    atoms = _parse_ca_atoms(TINY_PDB)
    assert len(atoms) == 7
    chains = {a["chain"] for a in atoms}
    assert chains == {"H", "L"}
    h1 = [a for a in atoms if a["chain"] == "H" and a["resseq"] == 1][0]
    assert h1["x"] == pytest.approx(10.0)
    assert h1["b"] == pytest.approx(90.0)


def test_parse_ca_skips_non_atom_records() -> None:
    pdb = "REMARK comment line\n" + TINY_PDB + "HETATM 999  C1  LIG A  10\n"
    atoms = _parse_ca_atoms(pdb)
    assert len(atoms) == 7   # only the ATOM CA records


def test_parse_ca_skips_non_ca_atoms() -> None:
    pdb = "ATOM      1  N   ALA H   1      10.000  10.000  10.000  1.00 90.00\n" + TINY_PDB
    atoms = _parse_ca_atoms(pdb)
    assert len(atoms) == 7   # the N atom isn't a CA → skipped


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_render_overview_returns_b64_png() -> None:
    out = render_structure(pdb_text=TINY_PDB, views=["overview"])
    assert "overview" in out
    png_b64 = out["overview"]
    raw = base64.b64decode(png_b64)
    # PNG magic bytes
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_with_highlights_does_not_crash() -> None:
    out = render_structure(pdb_text=TINY_PDB, highlight_residues=[2, 3], views=["overview", "cdr3"])
    assert set(out["views"]) == {"overview", "cdr3"}
    for view in ("overview", "cdr3"):
        raw = base64.b64decode(out[view])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_interface_with_two_chains() -> None:
    out = render_structure(pdb_text=TINY_PDB, views=["interface"])
    assert "interface" in out
    raw = base64.b64decode(out["interface"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_falls_back_to_overview_for_single_chain() -> None:
    single_chain = "\n".join(
        line for line in TINY_PDB.splitlines() if " H " in line or line.startswith("END")
    )
    out = render_structure(pdb_text=single_chain, views=["interface"])
    assert "interface" in out          # interface view degraded to overview internally


def test_render_rejects_empty_pdb() -> None:
    with pytest.raises(ValueError, match="No CA atoms"):
        render_structure(pdb_text="REMARK only metadata\nEND\n")


def test_color_by_plddt_off_uses_chain_colors() -> None:
    """Sanity: color_by_plddt=False should still produce a valid PNG."""
    out = render_structure(pdb_text=TINY_PDB, color_by_plddt=False)
    raw = base64.b64decode(out["overview"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
