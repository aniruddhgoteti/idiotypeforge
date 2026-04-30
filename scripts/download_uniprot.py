#!/usr/bin/env python
"""Download UniProt SwissProt human, convert to FASTA, build a BLAST DB.

Used by the off-target tool to ensure designed binders don't cross-react
with any human protein.

Source:
    https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/

The full human SwissProt subset is ~150 MB compressed, ~20 K reviewed entries.

Usage:
    python scripts/download_uniprot.py
    python scripts/download_uniprot.py --no-blast      # skip BLAST DB build
    python scripts/download_uniprot.py --dry-run
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import click

DEFAULT_OUT = Path("data/uniprot")

# SwissProt human FASTA — much smaller than the .dat file, easier to BLAST.
UNIPROT_HUMAN_FASTA_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=organism_id:9606+AND+reviewed:true"
    "&format=fasta"
    "&compressed=true"
)


def step(msg: str) -> None:
    click.echo(click.style("→ ", fg="blue") + msg)


def warn(msg: str) -> None:
    click.echo(click.style("! ", fg="yellow") + msg, err=True)


def ok(msg: str) -> None:
    click.echo(click.style("✓ ", fg="green") + msg)


def err(msg: str) -> None:
    click.echo(click.style("✗ ", fg="red") + msg, err=True)


def fetch(url: str, dest: Path, timeout: int = 600) -> None:
    """Stream-download `url` → `dest` with a tiny progress meter."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "IdiotypeForge/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = resp.headers.get("Content-Length")
        total_int = int(total) if total else None
        bytes_done = 0
        with dest.open("wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bytes_done += len(chunk)
                if total_int:
                    pct = 100 * bytes_done / total_int
                    click.echo(f"\r  {bytes_done / 1e6:.1f}/{total_int / 1e6:.1f} MB ({pct:.0f}%)",
                               nl=False)
                else:
                    click.echo(f"\r  {bytes_done / 1e6:.1f} MB", nl=False)
    click.echo()


def gunzip_to(src: Path, dest: Path) -> None:
    with gzip.open(src, "rb") as gz:
        with dest.open("wb") as out:
            shutil.copyfileobj(gz, out)


def build_blast_db(fasta_path: Path, db_prefix: Path) -> bool:
    """Run `makeblastdb -dbtype prot`. Requires NCBI BLAST+ on PATH."""
    if shutil.which("makeblastdb") is None:
        warn(
            "makeblastdb not found on PATH. Install BLAST+ from "
            "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/ "
            "or `brew install blast` / `sudo apt install ncbi-blast+`."
        )
        return False
    try:
        subprocess.run(
            [
                "makeblastdb",
                "-in", str(fasta_path),
                "-dbtype", "prot",
                "-out", str(db_prefix),
                "-title", "uniprot_human_swissprot",
            ],
            check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        err(f"makeblastdb failed: {e.stderr[:500]}")
        return False


@click.command()
@click.option("--no-blast", is_flag=True, help="Skip BLAST DB build.")
@click.option("--dry-run", is_flag=True, help="Show plan, don't download.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=str(DEFAULT_OUT))
def main(no_blast: bool, dry_run: bool, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fasta_gz = out / "uniprot_human_sprot.fasta.gz"
    fasta = out / "uniprot_human_sprot.fasta"
    db_prefix = out / "uniprot_human_swissprot"

    plan = {
        "out_dir": str(out),
        "url": UNIPROT_HUMAN_FASTA_URL,
        "fasta_gz": str(fasta_gz),
        "fasta": str(fasta),
        "build_blast_db": not no_blast,
        "blast_db_prefix": str(db_prefix),
    }
    import json
    click.echo(json.dumps(plan, indent=2))

    if dry_run:
        ok("dry-run complete.")
        sys.exit(0)

    step(f"downloading SwissProt human FASTA → {fasta_gz}")
    fetch(UNIPROT_HUMAN_FASTA_URL, fasta_gz)
    ok(f"downloaded {fasta_gz.stat().st_size / 1e6:.1f} MB")

    step("decompressing…")
    gunzip_to(fasta_gz, fasta)
    ok(f"FASTA: {fasta.stat().st_size / 1e6:.1f} MB")

    n_records = sum(1 for line in fasta.open() if line.startswith(">"))
    ok(f"{n_records:,} reviewed human proteins")

    if not no_blast:
        step("building BLAST DB…")
        if build_blast_db(fasta, db_prefix):
            ok(f"BLAST DB ready at {db_prefix}")
        else:
            warn("BLAST DB skipped or failed; FASTA still available for MMseqs2.")


if __name__ == "__main__":
    main()
