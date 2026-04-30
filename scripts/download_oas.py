#!/usr/bin/env python
"""Download a small healthy-paired subset of the Observed Antibody Space (OAS).

Local-first profile: pulls ~50K paired sequences (10–20 MB) — enough to build
a local MMseqs2 index for the off-target search tool during methodology
validation (Days 1–12).

GPU-phase profile (`--full`): pulls ~5–10 M sequences for the final fixture
runs and the live demo's off-target index.

OAS website: https://opig.stats.ox.ac.uk/webapps/oas/
Citation: Olsen et al. 2022 *Protein Sci*.

Usage:
    python scripts/download_oas.py                # local 50K sample
    python scripts/download_oas.py --full         # full 5–10 M paired
    python scripts/download_oas.py --build-index  # build MMseqs2 index
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

OAS_BASE = "https://opig.stats.ox.ac.uk/webapps/oas"
DEFAULT_OUT = Path("data/oas")


@click.command()
@click.option("--full", is_flag=True, help="Pull the full ~5–10M paired set (Day 13+).")
@click.option("--build-index", is_flag=True, help="Build MMseqs2 index after download.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=str(DEFAULT_OUT))
def main(full: bool, build_index: bool, out_dir: str) -> None:
    """Stub — implement on Day 3.

    Day-3 implementation:
        1. Query OAS with filters: species=human, type=paired, disease=None,
           subject!=child (for the healthy reference set).
        2. Download CSV.gz files for the sampled subjects.
        3. Concatenate IGH + IGK/L into a single FASTA per subject.
        4. If `--build-index`, run `mmseqs createdb` + `mmseqs createindex`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    click.echo(f"[stub] would download OAS to {out} (full={full}, index={build_index})")
    click.echo("[stub] implement on Day 3 — see scripts/download_oas.py docstring.")
    sys.exit(0)


if __name__ == "__main__":
    main()
