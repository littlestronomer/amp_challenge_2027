"""Download AMP training data into ``data/raw/<source>/``.

Run:  uv run --extra ml python scripts/fetch_data.py [--marlys-only | --dbaasp-only]
      uv run --extra ml python scripts/fetch_data.py --source dramp --url <direct-file-url>

Datasets:
  - MarLys (CC-0, ~102k peptides)   → data/raw/marlys/
  - DBAASP v3 (CC BY 4.0, MIC data) → data/raw/dbaasp/
  - Extra generative sources (DRAMP/APD/…) → data/raw/<source>/

Honesty note on URLs: most AMP databases serve downloads through click-mediated
pages (DRAMP V5's download page, APD3, dbAMP) without stable direct file URLs.
This fetcher therefore (a) tries the registered direct URLs where they exist,
(b) unzips automatically when a zip arrives, and (c) tells you exactly which
page to visit and which directory to drop the file into when automation can't
work. After fetching anything, run
``scripts/build_expanded_generative.py`` — it ingests every FASTA found under
``data/raw/**`` plus DBAASP sequences already processed.

After fetching, run ``scripts/build_datasets.py`` (MarLys + DBAASP curated
artifacts) and/or ``scripts/build_expanded_generative.py`` (merged corpus).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from amp_challenge_2027.config import RAW_DATA_DIR

MARLYS_MENDELEY_DATASET = "https://data.mendeley.com/datasets/w4hb5grjwb/3"
# Direct file URL changes between Mendeley record versions; override via
# --marlys-url. The default below is intentionally NOT fabricated.
MARLYS_URL_DEFAULT = ""

DBAASP_PEPTIDES_URL = "https://dbaasp.org/files/peptides.csv"  # adjust to current export
DBAASP_ACTIVITY_URL = "https://dbaasp.org/files/activity.csv"

# Human pages for click-mediated sources; drop the downloaded FASTA/zip under
# data/raw/<source>/ and build_expanded_generative.py picks it up.
MANUAL_SOURCES: dict[str, dict[str, str]] = {
    "dramp": {
        "page": "https://dramp.cpu-bioinfor.org/downloads/",
        "drop": "general-dataset FASTA/zip → data/raw/dramp/",
        "license": "CC BY 4.0",
    },
    "apd": {
        "page": "https://aps.unmc.edu/AP/",
        "drop": "sequence dump → data/raw/apd/",
        "license": "free for research; cite APD3 publication",
    },
    "dbamp": {
        "page": "https://awi.cuhk.edu.cn/dbAMP/",
        "drop": "FASTA export → data/raw/dbamp/",
        "license": "check site terms",
    },
}


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


def fetch_marlys(out_dir: Path | None = None, *, url: str | None = None) -> Path:
    """Download MarLys AMP database (or instruct manual download)."""
    out_dir = out_dir or (RAW_DATA_DIR / "marlys")
    dest = out_dir / "marlys.fasta"
    effective_url = url or MARLYS_URL_DEFAULT
    if not effective_url:
        print(
            "[fetch] MarLys: no direct URL registered (record URLs change per "
            f"version).\n"
            f"        Visit {MARLYS_MENDELEY_DATASET}\n"
            f"        then re-run with --marlys-url <direct FASTA link>, or place\n"
            f"        the file manually at {dest}"
        )
        return dest
    _download(effective_url, dest)
    _maybe_unzip(dest, out_dir)
    if dest.exists() and dest.stat().st_size > 0:
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
    parser = argparse.ArgumentParser(description="Download AMP training data.")
    parser.add_argument("--marlys-only", action="store_true")
    parser.add_argument("--dbaasp-only", action="store_true")
    parser.add_argument("--marlys-url", default=None, help="override MarLys FASTA URL")
    parser.add_argument(
        "--source", choices=sorted(MANUAL_SOURCES),
        help="print manual-download instructions for a click-mediated source",
    )
    parser.add_argument(
        "--url", default=None,
        help="direct file URL to download (used with --source or alone)",
    )
    args = parser.parse_args()

    if args.source:
        info = MANUAL_SOURCES[args.source]
        print(f"[fetch] {args.source}: open {info['page']}")
        print(f"        download the dataset, then {info['drop']}")
        print(f"        license: {info['license']}")
        if args.url:
            out_dir = RAW_DATA_DIR / args.source
            dest = _download(args.url, out_dir / Path(args.url).split("?")[0].rsplit("/", 1)[-1])
            _maybe_unzip(dest, out_dir)
            print(f"[fetch] saved under {out_dir}; build_expanded_generative.py will ingest it")
        return

    do_marlys = not args.dbaasp_only
    do_dbaasp = not args.marlys_only

    try:
        if do_marlys:
            fetch_marlys(url=args.marlys_url)
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
