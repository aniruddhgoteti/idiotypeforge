#!/usr/bin/env python
"""Download UniProt human SwissProt and build a BLAST DB.

Used by the off-target tool to ensure designed binders don't cross-react
with any human protein.

UniProt FTP: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/uniprot_sprot_human.dat.gz

~20K sequences, ~150 MB compressed. CPU-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

DEFAULT_OUT = Path("data/uniprot")


@click.command()
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=str(DEFAULT_OUT))
def main(out_dir: str) -> None:
    """Stub — implement on Day 3.

    Day-3 implementation:
        1. wget the gzipped human SwissProt
        2. gunzip and convert .dat → .fasta with biopython
        3. `makeblastdb -in human.fasta -dbtype prot -out uniprot_human_swissprot`
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    click.echo(f"[stub] would build UniProt human BLAST DB at {out}")
    click.echo("[stub] implement on Day 3.")
    sys.exit(0)


if __name__ == "__main__":
    main()
