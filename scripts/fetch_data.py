"""Download AMP training data into ``data/raw/<source>/``.

Run:  uv run --extra ml python scripts/fetch_data.py [--marlys-only | --dbaasp-only]
      uv run --extra ml python scripts/fetch_data.py --source dramp --url <direct-file-url>

Datasets:
  - MarLys (CC-0, ~102k peptides)   → data/raw/marlys/
  - DBAASP v3 (CC BY 4.0, MIC data) → data/raw/dbaasp/
  - Extra generative sources (DRAMP/APD/…) → data/raw/<source>/

Honesty note on URLs: most AMP databases serve downloads through click-mediated
pages (APD3, dbAMP) without stable direct file URLs. DRAMP 3.0 splits are the
exception: they download via ``download.php`` handlers (see
``DRAMP_DIRECT_SOURCES``; the Gram-split anchors are malformed upstream, so
those two PROBE candidate paths and fall back to exact manual instructions on
failure). This fetcher therefore (a) tries registered direct URLs where they
exist, (b) unzips automatically when a zip arrives, and (c) tells you exactly
which page to visit and which directory to drop the file into when automation
can't work. Successful downloads record provenance (url/timestamp/license/
sha256) into ``data/raw/sources.json`` — the skeleton of the submission's
training-data disclosure. After fetching anything, run
``scripts/build_expanded_generative.py`` or ``scripts/rebuild_corpus.py`` —
they ingest every FASTA found under ``data/raw/**`` plus DBAASP sequences
already processed.

After fetching, run ``scripts/build_datasets.py`` (MarLys + DBAASP curated
artifacts) and/or ``scripts/build_expanded_generative.py`` (merged corpus).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

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

# ---------------------------------------------------------------------------
# DRAMP 3.0 direct downloads (CC BY 4.0)
# Verified against the live downloads page: general + activity splits expose
# clean `download.php?filename=...` hrefs. The Gram-split anchors are
# MALFORMED upstream (positive has no href at all; negative hrefs truncate to
# `Anti-Gram-_amps.*`), so those paths are PROBES — a failed/non-FASTA
# response falls back to precise manual instructions, never to silent guesses.
# ---------------------------------------------------------------------------

# Verified live: relative anchors resolve under /downloads/ (the web-root
# handler 404s); the server also serves FASTA with a text/html content-type,
# so validators must inspect content, not headers.
DRAMP_DL_BASE = "https://dramp.cpu-bioinfor.org/downloads/download.php"
DRAMP_CITATION = "Kang et al., DRAMP 3.0, Sci Data 6:170 (2019); CC BY 4.0"


def _dramp_url(rel_path: str) -> str:
    return f"{DRAMP_DL_BASE}?filename={quote('download_data/DRAMP3.0_new/' + rel_path, safe='/')}"


DRAMP_DIRECT_SOURCES: dict[str, dict] = {
    "dramp-general": {
        "paths": ["general_amps.fasta"],
        "dest": "general_amps.fasta",
    },
    "dramp-antibacterial": {
        "paths": ["Antibacterial_amps.fasta"],
        "dest": "antibacterial_amps.fasta",
    },
    "dramp-grampos": {
        # probe: upstream anchor exists as visible text but its href never closes
        "paths": ["Anti-Gram-positive_amps.fasta"],
        "dest": "anti_gram_positive.fasta",
    },
    "dramp-gramneg": {
        # probe the full name first; upstream's own truncated path is the fallback
        "paths": ["Anti-Gram-negative_amps.fasta", "Anti-Gram-_amps.fasta"],
        "dest": "anti_gram_negative.fasta",
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


def _looks_fasta(path: Path) -> bool:
    """Cheap content check: rejects HTML error pages / empty downloads."""
    try:
        head = path.read_bytes()[:4096]
    except OSError:
        return False
    if not head.strip():
        return False
    lowered = head.lower()
    if b"<html" in lowered or b"<!doctype" in lowered:
        return False
    return any(line.startswith(b">") for line in head.splitlines())


def _record_provenance(
    source: str,
    url: str,
    path: Path,
    *,
    license_name: str,
    citation: str,
    registry_path: Path | None = None,
) -> Path:
    """Append {url, timestamp, license, sha256, size} to data/raw/sources.json.

    Deterministic JSON (sorted keys, fixed indent); re-running with the same
    file updates the timestamp only. This registry doubles as the skeleton of
    the full-track submission's training-data disclosure.
    """
    reg = Path(registry_path) if registry_path else (RAW_DATA_DIR / "sources.json")
    entry = {
        "url": url,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": license_name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "citation": citation,
    }
    try:
        registry: dict = json.loads(reg.read_text()) if reg.exists() else {}
    except json.JSONDecodeError:
        print(f"[fetch] WARNING: unreadable {reg}; starting a fresh registry")
        registry = {}
    key = f"{source}/{path.name}"
    prev = registry.get(key)
    if prev and prev.get("sha256") == entry["sha256"]:
        entry["retrieved_utc"] = prev.get("retrieved_utc", entry["retrieved_utc"])
    registry[key] = entry
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[fetch] provenance recorded → {reg} ({key})")
    return reg


def fetch_dramp(
    source: str, out_dir: Path | None = None, *, registry_path: Path | None = None
) -> Path | None:
    """Download one registered DRAMP split; manual fallback on any failure."""
    info = DRAMP_DIRECT_SOURCES[source]
    out_dir = out_dir or (RAW_DATA_DIR / "dramp")
    dest = out_dir / info["dest"]
    last_err = "no candidates"
    for rel in info["paths"]:
        url = _dramp_url(rel)
        part = out_dir / (info["dest"] + ".part")
        try:
            _download(url, part)
        except Exception as e:
            print(f"[fetch] probe failed ({rel}): {e}")
            last_err = str(e)
            continue
        if not _looks_fasta(part):
            size = part.stat().st_size
            print(
                f"[fetch] {rel}: response is not FASTA ({size} bytes; likely an upstream error page)"
            )
            last_err = f"{rel}: not FASTA"
            continue
        part.replace(dest)
        _record_provenance(
            source,
            url,
            dest,
            license_name="CC BY 4.0",
            citation=DRAMP_CITATION,
            registry_path=registry_path,
        )
        print(f"[fetch] DRAMP split at {dest}")
        return dest
    print(
        f"[fetch] {source}: automated download unavailable ({last_err}).\n"
        f"        Open {MANUAL_SOURCES['dramp']['page']}\n"
        f"        Grab one of these files manually:\n"
        + "".join(f"          - {rel}\n" for rel in info["paths"])
        + f"        Save it as {dest}"
    )
    return None


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
        "--source",
        choices=sorted(set(MANUAL_SOURCES) | set(DRAMP_DIRECT_SOURCES)),
        help="a click-mediated source (prints instructions) or a DRAMP split "
        "(downloads automatically)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="direct file URL to download (used with --source or alone)",
    )
    args = parser.parse_args()

    if args.source in DRAMP_DIRECT_SOURCES:
        fetch_dramp(args.source)
        return

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
