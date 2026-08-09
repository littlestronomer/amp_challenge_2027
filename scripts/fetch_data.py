"""Download MarLys AMP database and DBAASP v3 into ``data/raw/``.

Run:  uv run --extra ml python scripts/fetch_data.py [--marlys-only | --dbaasp-only]

Datasets:
  - MarLys (CC-0, ~102k peptides)  → data/raw/marlys/
  - DBAASP v3 (CC BY 4.0, MIC data) → data/raw/dbaasp/

These are explicit network fetches, never done at import time. After fetching,
run ``scripts/build_datasets.py`` (or ``amp_challenge_2027.data.build_datasets``)
to produce the curated parquet/csv artifacts in ``data/processed/``.

NOTE on DBAASP: the public download is a web export; column names vary. The
parser in ``amp_challenge_2027.data`` tolerates header aliasing. If the REST
API (https://dbaasp.org/api?page=rest) is preferred, a paginated fetcher can be
added here later — left as an extension point for your collaborator.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from amp_challenge_2027.config import RAW_DATA_DIR

MARLYS_ZENODO_URL = "https://data.mendeley.com/public-files/datasets/w4hb5grjwb/files/c7e1f9a0-3f5b-4f7e-9c8a-1b2c3d4e5f6a/file_downloaded"  # placeholder; see note
MARLYS_MENDELEY_DATASET = "https://data.mendeley.com/datasets/w4hb5grjwb/3"
DBAASP_PEPTIDES_URL = "https://dbaasp.org/files/peptides.csv"  # adjust to current export
DBAASP_ACTIVITY_URL = "https://dbaasp.org/files/activity.csv"


def _download(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` with a progress line; returns ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url}\n     → {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "amp-challenge-2027/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r     {read:>10} / {total} bytes ({pct:5.1f}%)", end="", flush=True)
        print()
    return dest


def fetch_marlys(out_dir: Path | None = None) -> Path:
    """Download MarLys AMP database.

    The Mendeley Data record (DOI 10.17632/w4hb5grjwb.3) hosts a FASTA dump.
    The exact file URL changes between record versions; if the direct link
    fails, visit ``MARLYS_MENDELEY_DATASET`` in a browser and point --marlys-url
    at the FASTA download.
    """
    out_dir = out_dir or (RAW_DATA_DIR / "marlys")
    dest = out_dir / "marlys.fasta"
    _download(MARLYS_ZENODO_URL, dest)
    # Some dumps are zipped; unzip if so.
    if _maybe_unzip(dest, out_dir):
        pass
    print(f"[fetch] MarLys FASTA at {dest}")
    return dest


def fetch_dbaasp(out_dir: Path | None = None) -> tuple[Path, Path]:
    """Download DBAASP peptide + activity CSV exports."""
    out_dir = out_dir or (RAW_DATA_DIR / "dbaasp")
    pep = _download(DBAASP_PEPTIDES_URL, out_dir / "peptides.csv")
    act = _download(DBAASP_ACTIVITY_URL, out_dir / "activity.csv")
    print(f"[fetch] DBAASP peptides at {pep}, activity at {act}")
    return pep, act


def _maybe_unzip(path: Path, out_dir: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if head[:2] != b"PK":
            return False
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
        print(f"[fetch] unzipped {path} into {out_dir}")
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MarLys + DBAASP training data.")
    parser.add_argument("--marlys-only", action="store_true")
    parser.add_argument("--dbaasp-only", action="store_true")
    parser.add_argument("--marlys-url", default=None, help="override MarLys FASTA URL")
    args = parser.parse_args()

    do_marlys = not args.dbaasp_only
    do_dbaasp = not args.marlys_only

    if args.marlys_url:
        global MARLYS_ZENODO_URL
        MARLYS_ZENODO_URL = args.marlys_url

    try:
        if do_marlys:
            fetch_marlys()
        if do_dbaasp:
            fetch_dbaasp()
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        print(
            "Note: dataset hosts occasionally change URLs. Visit the dataset "
            "pages directly and use --marlys-url if needed:\n"
            f"  MarLys: {MARLYS_MENDELEY_DATASET}\n"
            "  DBAASP: https://dbaasp.org",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
