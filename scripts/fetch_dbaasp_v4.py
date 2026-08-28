"""Fetch the DBAASP v4 REST API into the repo's raw CSV formats.

The working API (verified 2026-08-28; DBAASP API 4.0.1, OAS 3.0 — the docs
live in the swagger iframe of https://dbaasp.org/api?page=rest, spec at
``/v3/api-docs``):

    GET /peptides?limit=&offset=        → paginated index (NO activity data)
    GET /peptides/{peptideId}           → full card incl. ``targetActivities``
        [{targetSpecies: {name}, activityMeasureGroup: {name: "MIC"},
          concentration: "14.5", unit: {name: "µM"}, medium, ...}] and
        ``hemoliticCytotoxicActivities`` (kept for future safety labels).

Note the v2-era ``/api/v1?query=...`` endpoints from the old helper libraries
are DEAD (POST → 403, GET → empty) — that cost us a debugging round-trip.

Writes, in the exact formats the existing pipeline already parses:
    data/raw/dbaasp/peptides.csv   (id, sequence)            — monomers only
    data/raw/dbaasp/activity.csv   (peptide_id, target_organism, concentration, assay)
        where ``concentration`` is pre-normalized to "«µM» µM" so
        ``parse_mic_value`` cannot misread µg/ml or nM rows as µM.
    data/raw/dbaasp/hemolysis_raw.csv  — raw HC/EC50-style rows, side artifact

Politeness: bounded concurrency (default 4), retry with backoff, and a
resumable state file — reruns fetch only what is missing. Using the API
implies agreeing to DBAASP's data-usage policy; the citation requirement is
recorded in provenance by the caller.

Run (resumable; safe to re-run):
    uv run python scripts/fetch_dbaasp_v4.py            # ~25k detail calls
    # then: uv run python scripts/rebuild_corpus.py     # bootstraps mic.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from convert_dramp_xlsx import peptide_mw  # shared average-residue MW

BASE = "https://dbaasp.org"
UA = "amp-challenge-2027/0.1 (academic competition use; contact via repo)"
PAGE_SIZE = 200

# Units we can normalize to µM; anything else is counted and skipped.
_UNIT_TO_UM = {
    "µM": 1.0,
    "μM": 1.0,
    "uM": 1.0,
    "nM": 1e-3,
    "mM": 1e3,
    "µg/ml": None,
    "μg/ml": None,
    "ug/ml": None,
    "µg/mL": None,  # need MW
}
_MW_UNITS = {"µg/ml", "μg/ml", "ug/ml", "µg/mL"}


def _get(url: str, *, timeout: float = 30.0, retries: int = 4) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as e:
            last_err = e
            status = getattr(e, "code", None)
            if status in (404,):
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def index_all_peptides(*, page_size: int = PAGE_SIZE, pause: float = 0.2) -> list[dict]:
    """Page through the index; returns every record (monomers + complexes)."""
    out: list[dict] = []
    offset = 0
    total = None
    while True:
        payload = _get(f"{BASE}/peptides?limit={page_size}&offset={offset}")
        data = payload.get("data", [])
        total = payload.get("total", payload.get("totalCount", total))
        out.extend(data)
        offset += len(data)
        print(f"[dbaasp] index: {len(out)}/{total}", end="\r", flush=True)
        if not data or (total is not None and offset >= total):
            break
        time.sleep(pause)
    print(f"\n[dbaasp] index complete: {len(out)} records (total={total})")
    return out


_OP_ASCII = {"<": "<", ">": ">", "<=": "<=", ">=": ">=", "≤": "<=", "≥": ">="}


def concentration_to_um(conc: str, unit_name: str, mw: float) -> tuple[float | None, str]:
    """(concentration string, unit name) → (µM value, "«op»«val» µM" string).

    Unicode censoring (≤ ≥) is normalized to ASCII (<= >=) so the downstream
    ``parse_mic_value`` regex (which only knows [><]=?) keeps working.
    """
    raw = conc.strip().replace(",", ".")
    m = re.match(r"([<>≤≥]=?)?\s*(\d+(?:\.\d+)?)", raw)
    if not m:
        return None, ""
    op = _OP_ASCII.get(m.group(1) or "", "") if m.group(1) else ""
    num = float(m.group(2))
    factor = _UNIT_TO_UM.get(unit_name)
    if factor is None and unit_name not in _MW_UNITS:
        return None, ""
    val = num * factor if factor is not None else num * 1000.0 / mw
    return val, f"{op}{val:g} µM"


def parse_detail(detail: dict) -> tuple[list[dict], list[dict], dict]:
    """Detail card → (activity_rows, hemolysis_rows, sequence_info).

    Activity rows keep the RAW strain-level species string (the label builder
    maps genera; strain tokens are preserved for future strain-level heads).
    """
    monomers = detail.get("monomers") or []
    seqs = [m.get("sequence", "") for m in monomers if m.get("sequence")]
    sequence = detail.get("sequence") or (seqs[0] if len(seqs) == 1 else "")
    info = {
        "id": detail.get("id"),
        "dbaaspId": detail.get("dbaaspId", ""),
        "sequence": sequence,
        "complexity": (detail.get("complexity") or {}).get("name", ""),
        "n_terminus": detail.get("nTerminus") or "",
        "c_terminus": detail.get("cTerminus") or "",
    }

    activity_rows: list[dict] = []
    for a in detail.get("targetActivities") or []:
        if (a.get("activityMeasureGroup") or {}).get("name") != "MIC":
            continue
        species = (a.get("targetSpecies") or {}).get("name", "").strip()
        unit = ((a.get("unit") or {}).get("name") or "").strip()
        conc = str(a.get("concentration") or "").strip()
        if not species or not conc:
            continue
        val_um, normalized = concentration_to_um(conc, unit, mw=peptide_mw(sequence or "A"))
        if val_um is None:
            continue
        activity_rows.append(
            {
                "peptide_id": info["dbaaspId"] or info["id"],
                "target_organism": species,
                "concentration": normalized,
                "assay": f"{conc} {unit}".strip(),
            }
        )

    hemo_rows: list[dict] = []
    for h in detail.get("hemoliticCytotoxicActivities") or []:
        hemo_rows.append(
            {
                "peptide_id": info["dbaaspId"] or info["id"],
                "kind": h.get("activityType") or "",
                "target": json.dumps(
                    h.get("targetCells") or h.get("targetSpecies") or "", ensure_ascii=False
                ),
                "value": str(h.get("concentration") or ""),
                "unit": (h.get("unit") or {}).get("name", "")
                if isinstance(h.get("unit"), dict)
                else str(h.get("unit") or ""),
            }
        )
    return activity_rows, hemo_rows, info


def fetch_all(
    *,
    out_dir: Path,
    workers: int = 4,
    pause: float = 0.1,
    max_records: int | None = None,
) -> None:
    state_path = out_dir / ".dbaasp_fetch_state.jsonl"
    done: set[int] = set()
    if state_path.exists():
        with open(state_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
        print(f"[dbaasp] resuming: {len(done)} records already fetched")

    index = index_all_peptides()
    todo = [
        r
        for r in index
        if r.get("id") not in done and (r.get("sequence") or (r.get("monomers") or []))
    ]
    if max_records:
        todo = todo[:max_records]
    print(f"[dbaasp] detail fetch: {len(todo)} records (skipping {len(index) - len(todo)})")

    peptides_f = open(out_dir / "peptides.csv", "a", newline="")
    activity_f = open(out_dir / "activity.csv", "a", newline="")
    hemo_f = open(out_dir / "hemolysis_raw.csv", "a", newline="")
    pw = csv.DictWriter(peptides_f, fieldnames=["id", "sequence", "dbaasp_id", "complexity"])
    aw = csv.DictWriter(
        activity_f, fieldnames=["peptide_id", "target_organism", "concentration", "assay"]
    )
    hw = csv.DictWriter(hemo_f, fieldnames=["peptide_id", "kind", "target", "value", "unit"])
    if peptides_f.tell() == 0:
        pw.writeheader()
    if activity_f.tell() == 0:
        aw.writeheader()
    if hemo_f.tell() == 0:
        hw.writeheader()

    state_f = open(state_path, "a")

    def work(rec: dict) -> int:
        pid = rec["id"]
        detail = _get(f"{BASE}/peptides/{pid}")
        activities, hemo, info = parse_detail(detail)
        if info["sequence"]:
            pw.writerow(
                {
                    "id": info["id"],
                    "sequence": info["sequence"],
                    "dbaasp_id": info["dbaaspId"],
                    "complexity": info["complexity"],
                }
            )
        for row in activities:
            aw.writerow(row)
        for row in hemo:
            hw.writerow(row)
        time.sleep(pause)
        return pid

    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, rec): rec for rec in todo}
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                pid = fut.result()
                state_f.write(json.dumps({"id": pid}) + "\n")
                state_f.flush()
                n_done += 1
                if n_done % 500 == 0:
                    print(f"[dbaasp] {n_done}/{len(todo)} detail cards fetched", flush=True)
            except Exception as e:
                print(f"\n[dbaasp] id {rec.get('id')} failed: {e}", file=sys.stderr)
                if getattr(e, "args", None) and "404" in str(e):
                    state_f.write(json.dumps({"id": rec.get("id")}) + "\n")
    state_f.close()
    for f in (peptides_f, activity_f, hemo_f):
        f.close()
    print(f"[dbaasp] done: {n_done} new records; artifacts in {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch DBAASP v4 API → raw CSVs (resumable).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/dbaasp"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-records", type=int, default=None, help="cap detail calls (smoke testing)"
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fetch_all(out_dir=args.out_dir, workers=args.workers, max_records=args.max_records)


if __name__ == "__main__":
    main()
