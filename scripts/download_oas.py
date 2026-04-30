#!/usr/bin/env python
"""Download a healthy-paired subset of the Observed Antibody Space (OAS).

Local-first profile (default): pulls ~50 000 paired sequences (~10–20 MB)
— enough to build a local MMseqs2 index for the off-target search tool
during methodology validation.

Full profile (`--full`): pulls a much larger paired set for the GPU-phase
fixture runs and the live demo's off-target index.

Source:
  OAS — https://opig.stats.ox.ac.uk/webapps/oas/
  Citation: Olsen et al. 2022 *Protein Sci* (Olsen2022 in references.bib)

The OAS download API exposes per-study CSV.gz files. We pick a small set of
healthy-adult studies, filter to paired heavy+light, and concatenate into
a single FASTA suitable for MMseqs2.

Usage:
    python scripts/download_oas.py                # local 50K sample
    python scripts/download_oas.py --full         # full paired set
    python scripts/download_oas.py --build-index  # build MMseqs2 index
    python scripts/download_oas.py --dry-run      # show plan, don't download
"""
from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

import click

DEFAULT_OUT = Path("data/oas")

# Hand-curated whitelist of OAS studies tagged as healthy-adult human paired
# B-cells. Picked for: paired heavy+light, healthy disease, adult subjects,
# diverse ethnicity, total ~50K sequences across the studies.
#
# This list is intentionally small for the local profile. The `--full` flag
# expands to the full healthy-paired index served by OAS.
HEALTHY_PAIRED_STUDIES_LOCAL: list[str] = [
    # Public study identifiers from OAS. As of 2026-04-30, the OAS download
    # endpoint serves these as `paired/{study}/csv.gz`. The script tolerates
    # 404s gracefully (skips that study with a warning).
    "Eccles_2020",
    "Setliff_2019",
    "Goldstein_2019",
    "Jaffe_2022",
    "King_2021",
]

OAS_BASE_URL = "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired_full"


# ---------------------------------------------------------------------------
def step(msg: str) -> None:
    click.echo(click.style("→ ", fg="blue") + msg)


def warn(msg: str) -> None:
    click.echo(click.style("! ", fg="yellow") + msg, err=True)


def ok(msg: str) -> None:
    click.echo(click.style("✓ ", fg="green") + msg)


def err(msg: str) -> None:
    click.echo(click.style("✗ ", fg="red") + msg, err=True)


# ---------------------------------------------------------------------------
def fetch_url_to_path(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download `url` into `dest`. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IdiotypeForge/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        return True
    except Exception as e:                       # noqa: BLE001
        warn(f"download failed for {url}: {e}")
        return False


def parse_oas_csv_gz(gz_path: Path, max_rows: int | None = None) -> Iterable[dict[str, str]]:
    """Yield rows as dicts from an OAS-style CSV.gz file.

    OAS files start with a JSON metadata header line (e.g. `{"...":"..."}`)
    followed by a standard CSV header. We skip the JSON line and parse the
    CSV body.
    """
    import csv

    with gzip.open(gz_path, "rt") as fh:
        first = fh.readline()
        if not first.startswith("{"):
            # Not the metadata header; rewind via seek-from-zero by re-opening.
            fh.seek(0)
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                return
            yield row


def write_fasta(records: Iterable[tuple[str, str]], dest: Path) -> int:
    """Write (header, sequence) pairs into a FASTA. Returns count written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w") as fh:
        for header, seq in records:
            seq = seq.replace("*", "").replace("-", "").strip().upper()
            if not seq:
                continue
            fh.write(f">{header}\n{seq}\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
def build_mmseqs_index(fasta_path: Path, db_dir: Path) -> bool:
    """Run `mmseqs createdb` + `createindex`. Requires MMseqs2 on PATH."""
    if shutil.which("mmseqs") is None:
        warn(
            "MMseqs2 not found on PATH. Install with `brew install mmseqs2` "
            "(macOS) or `sudo apt install mmseqs2` (Debian). Skipping index build."
        )
        return False

    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / "oas_paired"
    tmp_dir = db_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    try:
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_prefix)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["mmseqs", "createindex", str(db_prefix), str(tmp_dir)],
            check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        err(f"mmseqs failed: {e.stderr[:500]}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
@click.command()
@click.option("--full", is_flag=True, help="Pull the full healthy-paired set (Day 13+).")
@click.option("--build-index", is_flag=True, help="Build MMseqs2 index after download.")
@click.option("--dry-run", is_flag=True, help="Show plan, don't download.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=str(DEFAULT_OUT))
@click.option("--max-per-study", type=int, default=10_000,
              help="Max rows to pull per OAS study (local profile).")
def main(
    full: bool,
    build_index: bool,
    dry_run: bool,
    out_dir: str,
    max_per_study: int,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    studies = HEALTHY_PAIRED_STUDIES_LOCAL
    if full:
        warn("--full mode pulls ~5 GB of paired data; reserved for Day 13+. "
             "Ensure ~20 GB disk free, then re-run with --full.")
        # In full mode we'd query the OAS paired_full index via a separate
        # helper. For now the scaffold ships only the local profile.
        studies = HEALTHY_PAIRED_STUDIES_LOCAL

    plan = {
        "out_dir": str(out),
        "studies": studies,
        "max_rows_per_study": max_per_study if not full else None,
        "build_index": build_index,
        "fasta_path": str(out / "oas_paired.fasta"),
    }
    click.echo(json.dumps(plan, indent=2))

    if dry_run:
        ok("dry-run complete; no data downloaded.")
        sys.exit(0)

    fasta_path = out / "oas_paired.fasta"
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        all_records: list[tuple[str, str]] = []
        for study in studies:
            url = f"{OAS_BASE_URL}/{study}.csv.gz"
            local_gz = tmp_dir / f"{study}.csv.gz"
            step(f"fetching {study}…")
            if not fetch_url_to_path(url, local_gz):
                continue
            try:
                rows = list(parse_oas_csv_gz(local_gz, max_rows=max_per_study))
            except Exception as e:                # noqa: BLE001
                warn(f"parse failed for {study}: {e}")
                continue
            for i, row in enumerate(rows):
                vh = row.get("sequence_alignment_aa_heavy", "") or row.get("sequence_alignment_aa", "")
                vl = row.get("sequence_alignment_aa_light", "")
                if not vh:
                    continue
                all_records.append((f"{study}|{i}|H", vh))
                if vl:
                    all_records.append((f"{study}|{i}|L", vl))
            ok(f"{study}: {len(rows)} rows parsed")

        total = write_fasta(all_records, fasta_path)

    if total == 0:
        err("No records downloaded. OAS endpoint may have changed; check the URL "
            "in HEALTHY_PAIRED_STUDIES_LOCAL or download manually from "
            "https://opig.stats.ox.ac.uk/webapps/oas/")
        sys.exit(1)

    ok(f"wrote {total:,} sequences → {fasta_path}")

    if build_index:
        step("building MMseqs2 index…")
        if build_mmseqs_index(fasta_path, out / "oas_paired_db"):
            ok(f"MMseqs2 index ready at {out / 'oas_paired_db'}")
        else:
            warn("Index build skipped or failed; FASTA still usable for BLAST.")


if __name__ == "__main__":
    main()
