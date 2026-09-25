"""Extract per-strain MIC annotations from the DRAMP 3.0 XLSX table.

DBAASP's bulk routes are closed to us (API POST → 403 from both our networks,
GET is a silent no-op, and the v3 site exposes no download UI — verified
in-browser on 2026-08-28). DRAMP 3.0 (CC BY 4.0) turns out to carry the same
kind of payload in its ``general_amps.xlsx``: a free-text ``Activity`` column
packed with per-strain annotations like::

    Gram-positive bacteria: Staphylococcus aureus ATCC6538P (MIC≤0.13 μg/ml),
    Enterococcus faecalis JCM 5803 (MIC=50.8 nM), Campylobacter jejuni
    (MIC=0.025-6.4 µg/ml)

This converter parses those clauses into the SAME schema as ``mic.csv``
(sequence, target_organism, mic_value_um, unit, assay, source_db) so
``build_ranking_labels.py --mic`` can consume it unchanged and the panel
classifier gets per-genus evidence without DBAASP.

Unit policy (everything normalized to µM):
  µg/ml family (µg|μg|ug|mg|ng per ml)  → via deterministic average-residue MW
  nM, pmol/ml (== nM)                    → /1000
  µM family, nmol/ml (== µM)             → direct
  AU/* (arbitrary units)                 → unusable, counted and skipped
Ranges ("0.025-6.4") take the UPPER bound (conservative for activity bands);
censoring (≤ ≥ < >) is dropped but the RAW clause is preserved in ``assay``.

Standard-library only (xlsx = zip of XML) — no new dependencies.

Run:
    uv run python scripts/convert_dramp_xlsx.py \
        --xlsx data/raw/dramp/general_amps.xlsx \
        --out data/processed/mic_dramp.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

from build_ranking_labels import peptide_mw

# ---------------------------------------------------------------------------
# XLSX reading (stdlib; flat single-sheet tables only)
# ---------------------------------------------------------------------------


def _col_index(ref: str) -> int:
    """'B3' → 1 (zero-based column index from the cell reference)."""
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Parse a flat one-sheet xlsx into dicts keyed by the header row.

    Real spreadsheets OMIT empty cells, so cells are placed by their ``r=``A1``
    column reference rather than by position.
    """
    with zipfile.ZipFile(path) as z:
        shared = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
        strings = re.findall(r"<t[^>]*>([^<]*)</t>", shared)
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8", "ignore")

    rows: list[list[str]] = []
    for row_xml in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
        cells: dict[int, str] = {}
        width = 0
        for m in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', row_xml, re.S):
            ref, attrs, body = m.groups()
            v = re.search(r"<v>([^<]*)</v>", body)
            if 't="s"' in attrs and v:
                cells[_col_index(ref)] = strings[int(v.group(1))]
            elif v:
                cells[_col_index(ref)] = v.group(1)
            else:
                cells[_col_index(ref)] = ""
            width = max(width, _col_index(ref) + 1)
        rows.append([cells.get(i, "") for i in range(width)])
    if not rows:
        return []
    header = rows[0]
    n_cols = len(header)
    return [dict(zip(header, (r + [""] * n_cols)[:n_cols])) for r in rows[1:] if any(r)]


# ---------------------------------------------------------------------------
# MIC clause extraction from Activity free text
# ---------------------------------------------------------------------------

_MIC_RE = re.compile(
    r"MIC\s*(?P<op>[<>=≤≥]{0,2})?\s*"
    r"(?P<val>\d+(?:[.,]\d+)?(?:\s*[-–~]\s*\d+(?:[.,]\d+)?)?)\s*"
    r"(?P<unit>µg/ml|μg/ml|ug/ml|mg/ml|ng/ml|µM|μM|uM|nM|pmol/ml|nmol/ml|AU/µg|AU/ug|AU/ml)",
    re.I,
)

# Panel genera: full binomials and the common abbreviations found in DRAMP
# text ("S. aureus", "E. coli", "Salmonella Enteritidis" with capitalized
# species epithets, etc.).
_GENUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"staphylococcus aureus|\bs\.\s*aureus", re.I), "Staphylococcus aureus"),
    (re.compile(r"escherichia coli|\be\.\s*coli", re.I), "Escherichia coli"),
    (re.compile(r"pseudomonas aeruginosa|\bp\.\s*aeruginosa", re.I), "Pseudomonas aeruginosa"),
    (re.compile(r"klebsiella pneumoni\S*|\bk\.\s*pneumoni\S*", re.I), "Klebsiella pneumoniae"),
    (re.compile(r"acinetobacter baumannii|\ba\.\s*baumannii", re.I), "Acinetobacter baumannii"),
    (re.compile(r"enterobacter cloacae|\be\.\s*cloacae", re.I), "Enterobacter cloacae"),
    (re.compile(r"salmonella", re.I), "Salmonella"),
    (re.compile(r"enterococcus faecalis|\be\.\s*faecalis", re.I), "Enterococcus faecalis"),
    (re.compile(r"enterococcus faecium|\be\.\s*faecium", re.I), "Enterococcus faecium"),
    (re.compile(r"bacillus subtilis|\bb\.\s*subtilis", re.I), "Bacillus subtilis"),
]


def match_panel_genus(window: str) -> str:
    """Map a text window to a DBAASP-style genus name, '' if no panel match."""
    for pattern, genus in _GENUS_PATTERNS:
        if pattern.search(window):
            return genus
    return ""


def _normalize_unit(unit: str) -> str:
    u = unit.strip().lower().replace("μ", "µ")
    if u in ("ug/ml", "µg/ml"):
        return "ug_ml"
    if u in ("mg/ml",):
        return "mg_ml"
    if u in ("ng/ml",):
        return "ng_ml"
    if u == "nm":
        return "nM"
    if u == "µm":
        return "uM"
    if u == "pmol/ml":
        return "pmol_ml"
    if u == "nmol/ml":
        return "nmol_ml"
    return u


def convert_to_um(value: float, unit: str, mw: float) -> tuple[float | None, str]:
    """Normalized-unit value → µM. Returns (value_or_None, note)."""
    u = _normalize_unit(unit)
    if u == "uM":
        return value, ""
    if u == "nM" or u == "pmol_ml":
        return value / 1000.0, ""
    if u == "nmol_ml":
        return value, ""
    if u == "ug_ml":
        return value * 1000.0 / mw, ""
    if u == "ng_ml":
        return value / mw, ""
    if u == "mg_ml":
        return value * 1_000_000.0 / mw, ""
    return None, f"unusable unit '{unit}'"


def extract_mic_entries(activity_text: str, sequence: str) -> list[dict]:
    """Activity text → [{organism, value_um, raw_clause}]. One per MIC clause
    whose preceding window names a panel genus."""
    mw = peptide_mw(sequence)
    entries: list[dict] = []
    for match in _MIC_RE.finditer(activity_text):
        window = activity_text[max(0, match.start() - 120) : match.start()]
        window = re.sub(r"\([^)]*\)", " ", window)  # drop earlier MIC parens
        # Only the CURRENT clause's species text — cut at the last delimiter
        # so clause N doesn't inherit clause N-1's genus.
        window = re.split(r"[;,]", window)[-1]
        genus = match_panel_genus(window)
        if not genus:
            continue
        raw_val = match.group("val").replace(",", ".")
        if re.search(r"[-–~]", raw_val):
            value = float(raw_val.split(re.search(r"[-–~]", raw_val).group(0))[-1])
            range_note = "range-upper"
        else:
            value, range_note = float(raw_val), ""
        val_um, note = convert_to_um(value, match.group("unit"), mw)
        if val_um is None:
            continue
        entries.append(
            {
                "organism": genus,
                "value_um": val_um,
                "raw_clause": match.group(0).strip(),
                "note": " ".join(n for n in (range_note, note) if n),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DRAMP xlsx → mic_dramp.csv")
    parser.add_argument("--xlsx", type=Path, default=Path("data/raw/dramp/general_amps.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/mic_dramp.csv"))
    args = parser.parse_args(argv)

    if not args.xlsx.exists():
        print(f"[dramp-xlsx] missing: {args.xlsx}; fetch it first (see runbook)", file=sys.stderr)
        sys.exit(1)

    rows = read_xlsx_rows(args.xlsx)
    print(f"[dramp-xlsx] {len(rows)} records in {args.xlsx.name}")

    out_rows: list[dict] = []
    stats = {
        "no_activity_text": 0,
        "clauses": 0,
        "rows_without_panel_genus": 0,
        "nonlinear_or_modified": 0,
    }
    unmodified = {"", "free", "none", "-"}
    for r in rows:
        seq = (r.get("Sequence") or "").strip().upper()
        # The per-strain MIC clauses live in Target_Organism (despite the
        # column's name), not in Activity — verified against the real 3.0
        # export (2026-08-28).
        mic_text = (r.get("Target_Organism") or "").strip() or (r.get("Activity") or "").strip()
        if not seq or not mic_text:
            stats["no_activity_text"] += 1
            continue
        # Real column values: 'Linear', 'Linear ' (trailing space!), 'Cyclic',
        # 'Free', 'Amidation', 'Not included yet' — strip + skip only definite
        # cyclic/branched or definite modifications; unknowns stay.
        linearity = (r.get("Linear/Cyclic/Branched") or "").strip().lower()
        if ("cyclic" in linearity or "branched" in linearity) and "possibly" not in linearity:
            stats["nonlinear_or_modified"] += 1
            continue
        n_mod = (r.get("N-terminal_Modification") or "").strip().lower()
        c_mod = (r.get("C-terminal_Modification") or "").strip().lower()
        n_known_mod = n_mod not in unmodified and "not included" not in n_mod
        c_known_mod = c_mod not in unmodified and "not included" not in c_mod
        if n_known_mod or c_known_mod:
            stats["nonlinear_or_modified"] += 1
            continue
        entries = extract_mic_entries(mic_text, seq)
        stats["clauses"] += len(entries)
        if _MIC_RE.search(mic_text) and not entries:
            stats["rows_without_panel_genus"] += 1
        for e in entries:
            out_rows.append(
                {
                    "sequence": seq,
                    "target_organism": e["organism"],
                    "mic_value_um": f"{e['value_um']:.4f}",
                    "unit": "uM",
                    "assay": f"DRAMP:{e['raw_clause']}",
                    "source_db": "DRAMP",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "target_organism",
                "mic_value_um",
                "unit",
                "assay",
                "source_db",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    per_genus: dict[str, int] = {}
    seqs: set[str] = set()
    for r in out_rows:
        per_genus[r["target_organism"]] = per_genus.get(r["target_organism"], 0) + 1
        seqs.add(r["sequence"])
    print(f"[dramp-xlsx] stats: {stats}")
    print(f"[dramp-xlsx] wrote {len(out_rows)} rows / {len(seqs)} sequences → {args.out}")
    print("[dramp-xlsx] per-genus rows:", dict(sorted(per_genus.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
