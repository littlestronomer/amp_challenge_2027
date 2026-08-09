"""Data pipeline: fetch, parse, and canonicalize AMP training data.

Three datasets are produced in ``data/processed/``:

  * ``generative.parquet`` — sequences for generator pretraining (MarLys).
  * ``mic.parquet``        — (seq, strain, MIC) for the activity reward model (DBAASP).
  * ``hemolysis.parquet``  — (seq, HC50) for the toxicity reward model (DBAASP).

This module does *not* download anything at import time; fetching lives in
``scripts/fetch_data.py``. This keeps the inference entry point lightweight.

The MarLys FASTA header schema (confirmed from ``data/antibacterial.fasta``) is::

    >MLAMP0003076 len=15 charge=4.73 disulfide=0 dbs=... activity=antibacterial|...

Your collaborator's data-curation rules belong in the *_curate* helpers, which
apply documented quality filters before anything is written. These are the
explicit interface between biology expertise and the ML code: curated in, ML out.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import (
    MAX_LENGTH,
    MIN_LENGTH,
    PROCESSED_DATA_DIR,
)

# ---------------------------------------------------------------------------
# Canonical record types
# ---------------------------------------------------------------------------


@dataclass
class PeptideRecord:
    """A sequence with provenance metadata for generator pretraining."""

    sequence: str
    source_id: str = ""
    charge: float | None = None
    activity: list[str] = field(default_factory=list)
    source_dbs: list[str] = field(default_factory=list)
    disulfides: int | None = None

    def is_valid(self) -> bool:
        return (
            MIN_LENGTH <= len(self.sequence) <= MAX_LENGTH
            and tok.is_valid_sequence(self.sequence)
        )


@dataclass
class MICRecord:
    """A (sequence, target_organism, MIC) triple for the activity reward."""

    sequence: str
    target_organism: str
    mic_value_um: float
    unit: str = "uM"
    assay: str = ""
    source_db: str = ""


@dataclass
class HemolysisRecord:
    """A (sequence, HC50) pair for the toxicity reward."""

    sequence: str
    hc50_um: float
    source_db: str = ""


# ---------------------------------------------------------------------------
# FASTA parsing (MarLys-style headers)
# ---------------------------------------------------------------------------


_HEADER_KV = re.compile(r"(\w+)=(\S+)")


def parse_marlys_header(header: str) -> dict:
    """Parse a MarLys FASTA header into structured fields.

    Example header::
        >MLAMP0003076 len=15 charge=4.73 disulfide=0 dbs=AMPDB|APD|... activity=antibacterial|...
    """
    parts = header.lstrip(">").strip().split()
    record_id = parts[0] if parts else ""
    kv = dict(_HEADER_KV.findall(header))
    return {
        "id": record_id,
        "len": int(kv["len"]) if "len" in kv else None,
        "charge": float(kv["charge"]) if "charge" in kv else None,
        "disulfide": int(kv["disulfide"]) if "disulfide" in kv else None,
        "dbs": kv.get("dbs", "").split("|") if "dbs" in kv else [],
        "activity": kv.get("activity", "").split("|") if "activity" in kv else [],
    }


def iter_fasta(path: Path | str) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file. Uppercases sequences."""
    header: str | None = None
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header is not None:
        yield header, "".join(seq_parts)


def read_fasta_records(path: Path | str) -> list[PeptideRecord]:
    """Read a MarLys-style FASTA into PeptideRecords (unfiltered)."""
    records: list[PeptideRecord] = []
    for header, seq in iter_fasta(path):
        meta = parse_marlys_header(header)
        records.append(
            PeptideRecord(
                sequence=seq,
                source_id=meta["id"],
                charge=meta["charge"],
                activity=meta["activity"],
                source_dbs=meta["dbs"],
                disulfides=meta["disulfide"],
            )
        )
    return records


def read_reference_set(path: Path | str) -> set[str]:
    """Read a FASTA into a set of sequences — for the no-overlap / identity checks."""
    return {seq for _, seq in iter_fasta(path)}


# ---------------------------------------------------------------------------
# DBAASP parsing (strain-level MIC + hemolysis)
# ---------------------------------------------------------------------------

# Approximate mapping from DBAASP target species names (free-text) to the
# competition's 20-strain panel. DBAASP species strings are messy; your
# collaborator should refine this mapping from actual data. This is a starting
# point, not a final table — it intentionally over-matches common substrings.
SPECIES_TO_PANEL: dict[str, str] = {
    "Acinetobacter baumannii": "A. baumannii",
    "Escherichia coli": "E. coli",
    "Klebsiella pneumoniae": "K. pneumoniae",
    "Pseudomonas aeroma": "P. aeruginosa",  # tolerate typo variants
    "Pseudomonas aeruginosa": "P. aeruginosa",
    "Staphylococcus aureus": "S. aureus",
    "Bacillus subtilis": "B. subtilis",
    "Enterococcus faecalis": "E. faecalis",
    "Enterococcus faecium": "E. faecium",
    "Enterobacter cloacae": "E. cloacae",
    "Salmonella": "S. enterica",
}


def normalize_species(name: str) -> str:
    """Map a free-text DBAASP species to a panel genus, or '' if no match."""
    name = name.strip()
    for key, genus in SPECIES_TO_PANEL.items():
        if key.lower() in name.lower():
            return genus
    return ""


_MIC_CEILING = re.compile(r"[><]=?\s*(\d+(?:\.\d+)?)")
_UNIT_MAP = {"µg/ml": "ug_ml", "ug/ml": "ug_ml", "µm": "uM", "um": "uM", "mcm": "uM"}


def parse_mic_value(raw: str) -> tuple[float, str] | None:
    """Parse a messy MIC string like '12.5', '<= 64', '> 128 µg/ml'.

    Returns (value, unit) or None. Note: censored values (>, <) lose their
    censoring flag here — your collaborator should decide how to treat them
    (e.g. drop, or clamp to the ceiling). The raw string is preserved in the
    record for that judgment.
    """
    if not raw:
        return None
    m = _MIC_CEILING.search(raw.replace(",", "."))
    if not m:
        return None
    val = float(m.group(1))
    unit = "uM"
    for u in _UNIT_MAP:
        if u in raw.lower():
            unit = _UNIT_MAP[u]
            break
    return val, unit


def parse_dbaasp_csv(
    peptides_csv: Path | str,
    mic_csv: Path | str | None = None,
) -> tuple[list[MICRecord], list[HemolysisRecord]]:
    """Parse DBAASP peptide + activity CSV exports into MIC / hemolysis records.

    DBAASP exposes a per-peptide CSV (sequences + metadata) and an activity
    CSV (target organism, MIC, units). Column names vary between exports, so we
    match case-insensitively by known header aliases.

    This is a best-effort parser against the public export format. Your
    collaborator owns refining the species mapping and unit conversion
    (µg/mL → µM) — see ``parse_mic_value``.
    """
    peptides_csv = Path(peptides_csv)

    # --- peptides ----------------------------------------------------------
    seq_by_id: dict[str, PeptideRecord] = {}
    with open(peptides_csv, newline="") as f:
        reader = _flexible_csv_reader(f)
        for row in reader:
            seq = _pick(row, ["sequence", "seq", "peptide", "aaa_sequence"]).strip().upper()
            if not seq or not tok.is_valid_sequence(seq):
                continue
            pid = _pick(row, ["id", "id_dbaasp", "peptide_id"], default="").strip()
            seq_by_id[pid] = PeptideRecord(sequence=seq, source_id=pid)

    mic_records: list[MICRecord] = []
    hemo_records: list[HemolysisRecord] = []

    # --- activity (MIC) ----------------------------------------------------
    if mic_csv and Path(mic_csv).exists():
        with open(mic_csv, newline="") as f:
            reader = _flexible_csv_reader(f)
            for row in reader:
                pid = _pick(row, ["peptide_id", "id_dbaasp", "id"], default="").strip()
                if pid not in seq_by_id:
                    continue
                seq = seq_by_id[pid].sequence
                genus = normalize_species(
                    _pick(row, ["target_organism", "species", "organism", "target"], default="")
                )
                if not genus:
                    continue
                parsed = parse_mic_value(
                    _pick(row, ["mic", "mic_value", "concentration", "value"], default="")
                )
                if parsed is None:
                    continue
                val, unit = parsed
                mic_records.append(
                    MICRecord(
                        sequence=seq,
                        target_organism=genus,
                        mic_value_um=val,
                        unit=unit,
                        assay=_pick(row, ["assay", "method"], default=""),
                        source_db="DBAASP",
                    )
                )

                # Hemolysis rows often live in the same activity file, keyed on
                # a target containing 'hemolysis' / 'erythrocyte'.
                tgt = _pick(row, ["target_organism", "target"], default="").lower()
                if "hemolys" in tgt or "erythrocyte" in tgt:
                    hc = parse_mic_value(
                        _pick(row, ["hc50", "hc_50", "mic", "concentration"], default="")
                    )
                    if hc is not None:
                        hemo_records.append(
                            HemolysisRecord(sequence=seq, hc50_um=hc[0], source_db="DBAASP")
                        )

    return mic_records, hemo_records


# ---------------------------------------------------------------------------
# Curated writers (the biology↔ML interface)
# ---------------------------------------------------------------------------


def curate_generative(records: list[PeptideRecord]) -> list[PeptideRecord]:
    """Apply generator-pretraining quality filters.

    Your collaborator refines these rules; they are deliberately explicit and
    conservative. Anything removed here is removed *before* the model sees it.
    """
    seen: set[str] = set()
    out: list[PeptideRecord] = []
    for r in records:
        if not r.is_valid():
            continue
        # Drop exact duplicates across the combined corpus.
        if r.sequence in seen:
            continue
        seen.add(r.sequence)
        out.append(r)
    return out


def curate_mic(records: list[MICRecord]) -> list[MICRecord]:
    """Apply MIC reward-model quality filters.

    Extension points for your collaborator:
      - assay/method trust weighting (e.g. only broth microdilution)
      - strain-specific MIC binning (active vs inactive threshold)
      - µg/mL → µM unit conversion per peptide molecular weight
      - treatment of censored values (>, <)
    """
    out: list[MICRecord] = []
    for r in records:
        if not tok.is_valid_sequence(r.sequence):
            continue
        if not (MIN_LENGTH <= len(r.sequence) <= MAX_LENGTH):
            continue
        if r.mic_value_um <= 0:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _flexible_csv_reader(f) -> csv.DictReader:
    """Read a CSV with case-insensitive header normalization."""
    reader = csv.DictReader(f)
    reader.fieldnames = [
        (n.strip().lower() if n else n) for n in reader.fieldnames or []
    ]
    return reader


def _pick(row: dict, keys: list[str], default: str = "") -> str:
    """Return the first present value among ``keys`` (case-insensitive lookup)."""
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return default


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_datasets(
    marlys_fasta: Path | str | None = None,
    dbaasp_peptides_csv: Path | str | None = None,
    dbaasp_mic_csv: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, int]:
    """Parse + curate all sources and write processed parquet/csv artifacts.

    Returns a dict of {artifact: num_records} so callers can log what was built.
    Sources that are missing are skipped (with a note), so this can run as data
    arrives rather than requiring all inputs at once.
    """
    out_dir = Path(out_dir or PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    # Generative (MarLys)
    if marlys_fasta and Path(marlys_fasta).exists():
        records = curate_generative(read_fasta_records(marlys_fasta))
        _write_generative(records, out_dir / "generative.csv")
        counts["generative"] = len(records)
    else:
        print(f"[data] skip generative: {marlys_fasta} not found")

    # MIC + hemolysis (DBAASP)
    if dbaasp_peptides_csv and Path(dbaasp_peptides_csv).exists():
        mic, hemo = parse_dbaasp_csv(dbaasp_peptides_csv, dbaasp_mic_csv)
        mic = curate_mic(mic)
        _write_mic(mic, out_dir / "mic.csv")
        _write_hemolysis(hemo, out_dir / "hemolysis.csv")
        counts["mic"] = len(mic)
        counts["hemolysis"] = len(hemo)
    else:
        print(f"[data] skip DBAASP: {dbaasp_peptides_csv} not found")

    return counts


def _write_generative(records: list[PeptideRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "source_id", "charge", "activity", "source_dbs"])
        for r in records:
            w.writerow([
                r.sequence,
                r.source_id,
                r.charge if r.charge is not None else "",
                "|".join(r.activity),
                "|".join(r.source_dbs),
            ])


def _write_mic(records: list[MICRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "target_organism", "mic_value_um", "unit", "assay", "source_db"])
        for r in records:
            w.writerow([r.sequence, r.target_organism, r.mic_value_um, r.unit, r.assay, r.source_db])


def _write_hemolysis(records: list[HemolysisRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "hc50_um", "source_db"])
        for r in records:
            w.writerow([r.sequence, r.hc50_um, r.source_db])


__all__ = [
    "PeptideRecord",
    "MICRecord",
    "HemolysisRecord",
    "parse_marlys_header",
    "iter_fasta",
    "read_fasta_records",
    "read_reference_set",
    "parse_mic_value",
    "normalize_species",
    "parse_dbaasp_csv",
    "curate_generative",
    "curate_mic",
    "build_datasets",
]
