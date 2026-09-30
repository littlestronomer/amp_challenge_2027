"""Observation-preserving inputs for conditional-generator research.

This module never edits the deployed training data or interprets missing assays as
negative labels.  Raw snapshots and prepared datasets are immutable and resumable.
Only explicitly compatible chemical identities supervise the competition molecule.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

SCHEMA_VERSION = 1
AA = set("ACDEFGHIKLMNPQRSTVWY")
DBAASP = "https://dbaasp.org"
DRAMP_URL = ("https://dramp.cpu-bioinfor.org/downloads/download.php?filename="
             "download_data/DRAMP3.0_new/general_amps.xlsx")
HEMOLYTIK_RECORD = "https://zenodo.org/api/records/19699377"
SOURCES = {
    "marlys": {"url": "https://data.mendeley.com/datasets/w4hb5grjwb/3",
               "citation": "MarLys, Mendeley Data version 3",
               "terms": "Mendeley version 3 record declares CC0 1.0 (verified 2026-09-25). Aggregated source databases retain their own terms; no assay labels inferred from membership."},
    "dbaasp": {"url": "https://dbaasp.org/api?page=rest", "citation": "DBAASP database",
               "terms_url": "https://www.dbaasp.org/terms-and-conditions",
               "terms": "Current HTML policy permits access, copying, adaptation and redistribution with public DBAASP acknowledgement; cite Pirtskhalava et al., NAR 49(D1):D288-D297 (2021), doi:10.1093/nar/gkaa991. Verified 2026-09-25. Legacy API-linked PDF also contains conflicting non-distribution wording; retain both references when reviewing redistribution.",
               "legacy_terms_url": "https://dbaasp.org/docs/DBAASP_Terms_And_Conditions.pdf"},
    "dramp": {"url": "https://dramp.cpu-bioinfor.org/downloads/", "citation": "DRAMP database",
              "terms": "Downloads page states CC BY 4.0; recorded download time identifies this snapshot, not legacy URL directory."},
    "hemolytik": {"url": "https://zenodo.org/records/19699377", "citation": "Hemolytik 2",
                  "terms": "UNRESOLVED: Zenodo description states CC BY 4.0; machine-readable metadata declares GPL-3.0; repository GPL license and README MIT/noncommercial notices conflict. Default disabled; inclusion is not release-license approval."},
}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def _immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable input/output differs: {path}; use a new directory")
    else:
        _atomic(path, data)


def _get(url: str, *, retries: int = 4) -> bytes:
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "amp-challenge-2027 conditional-research/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError) as exc:
            error = exc
            if getattr(exc, "code", None) in (400, 401, 403, 404):
                raise
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from error


def _artifact(root: Path, rel: str, source: str, url: str, fetch: Callable[[], bytes]) -> dict:
    """A sidecar permits verified reuse after an interrupted full snapshot fetch."""
    path = root / rel
    sidecar = path.with_name(path.name + ".source.json")
    if sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["url"] != url or record["source"] != source:
            raise ValueError(f"Snapshot source identity changed: {path}")
        if not path.exists():
            data = fetch()
            if _sha(data) != record["sha256"]:
                raise ValueError(f"Interrupted snapshot source changed: {path}; use a new directory")
            _atomic(path, data)
        if _sha(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {path}")
        return record
    if path.exists():
        raise ValueError(f"Unverified snapshot artifact already exists: {path}")
    data = fetch()
    record = {"path": rel, "source": source, "url": url, "retrieved_utc": _utc(),
              "sha256": _sha(data), "size_bytes": len(data), **SOURCES[source]}
    # Keep artifact URL separate from the source's landing-page URL.
    record["url"] = url
    _atomic(sidecar, _json_bytes(record))
    _atomic(path, data)
    return record


def verify_snapshot(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported conditional snapshot schema")
    for record in manifest["artifacts"]:
        rel = Path(record["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe artifact path in snapshot manifest")
        path = root / rel
        if not path.exists() or _sha(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {path}")
    return manifest


def fetch_sources(out: Path, *, marlys: Path | None = None, dbaasp_dir: Path | None = None,
                  dramp: Path | None = None, include_hemolytik: bool = False,
                  hemolytik: Path | None = None, workers: int = 4,
                  max_records: int | None = None, opener: Callable[[str], bytes] | None = None) -> dict:
    """Create a dated raw snapshot. Network requests occur only for missing data.

    ``marlys`` must point at a frozen CSV/FASTA: this avoids silently substituting a
    later Mendeley corpus. ``dbaasp_dir`` imports full JSON cards (never old CSVs).
    ``max_records`` explicitly marks a network snapshot incomplete for smoke tests.
    ``opener`` is an injectable byte-returning GET function for offline tests.
    """
    out = Path(out)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if marlys is None or not Path(marlys).is_file():
        raise ValueError("Supply the frozen MarLys CSV/FASTA with --marlys; recover version 3 from " + SOURCES["marlys"]["url"])
    if hemolytik is not None and not include_hemolytik:
        raise ValueError("Hemolytik is disabled by default; explicitly enable research import")
    opener = opener or _get
    out.mkdir(parents=True, exist_ok=True)
    recipe = {"marlys": str(Path(marlys).resolve()), "marlys_sha256": _sha(Path(marlys).read_bytes()),
              "dbaasp_dir": str(Path(dbaasp_dir).resolve()) if dbaasp_dir else None,
              "dbaasp_local_hashes": {str(p.relative_to(dbaasp_dir)): _sha(p.read_bytes()) for p in sorted(Path(dbaasp_dir).rglob("*.json"))} if dbaasp_dir else None,
              "dramp": str(Path(dramp).resolve()) if dramp else None,
              "dramp_sha256": _sha(Path(dramp).read_bytes()) if dramp else None,
              "hemolytik": str(Path(hemolytik).resolve()) if hemolytik else None,
              "hemolytik_sha256": _sha(Path(hemolytik).read_bytes()) if hemolytik else None,
              "include_hemolytik": include_hemolytik, "max_records": max_records}
    _immutable(out / "recipe.json", _json_bytes(recipe))
    if (out / "manifest.json").exists():
        return verify_snapshot(out)
    records = []
    suffix = Path(marlys).suffix or ".csv"
    records.append(_artifact(out, "marlys/corpus" + suffix, "marlys", str(Path(marlys).resolve()), Path(marlys).read_bytes))
    complete = max_records is None
    if dbaasp_dir:
        cards = []
        for path in sorted(Path(dbaasp_dir).rglob("*.json")):
            if path.name.endswith(".source.json"):
                continue
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and "id" in payload and ("sequence" in payload or "monomers" in payload):
                cards.append((str(payload["id"]), path))
        if not cards:
            raise ValueError("--dbaasp-dir contains no full JSON cards; legacy CSV exports omit required assay metadata")
        for pid, path in cards[:max_records]:
            if not re.fullmatch(r"\d+", pid):
                raise ValueError("Invalid DBAASP card id")
            records.append(_artifact(out, f"dbaasp/cards/{pid}.json", "dbaasp", str(path.resolve()), path.read_bytes))
        # Local cards are a declared corpus, not a claim to be the entire database.
        complete = max_records is None
    else:
        ids, offset = [], 0
        while True:
            url = f"{DBAASP}/peptides?limit=200&offset={offset}"
            rel = f"dbaasp/index/{offset:08d}.json"
            records.append(_artifact(out, rel, "dbaasp", url, lambda url=url: opener(url)))
            payload = json.loads((out / rel).read_text())
            page = payload.get("data", [])
            if not isinstance(page, list):
                raise ValueError("Unexpected DBAASP index schema: data must be a list")
            ids.extend(str(item["id"]) for item in page)
            offset += len(page)
            total = payload.get("total", payload.get("totalCount"))
            if not page or (total is not None and offset >= int(total)):
                break
            if max_records is not None and len(ids) >= max_records:
                break
        ids = list(dict.fromkeys(ids))[:max_records]
        if not ids:
            raise ValueError("DBAASP returned no peptide IDs")
        def fetch_card(pid):
            if not re.fullmatch(r"\d+", pid):
                raise ValueError("Invalid DBAASP index id")
            url = f"{DBAASP}/peptides/{pid}"
            return _artifact(out, f"dbaasp/cards/{pid}.json", "dbaasp", url, lambda: opener(url))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = [executor.submit(fetch_card, pid) for pid in ids]
            for i, future in enumerate(as_completed(pending), 1):
                records.append(future.result())
                if i % 200 == 0:
                    print(f"[conditional-data] full DBAASP cards {i}/{len(ids)}", flush=True)
    records.append(_artifact(out, "dramp/general_amps.xlsx", "dramp",
                             str(Path(dramp).resolve()) if dramp else DRAMP_URL,
                             Path(dramp).read_bytes if dramp else lambda: opener(DRAMP_URL)))
    if include_hemolytik:
        if hemolytik:
            records.append(_artifact(out, "hemolytik/table" + Path(hemolytik).suffix, "hemolytik",
                                     str(Path(hemolytik).resolve()), Path(hemolytik).read_bytes))
        else:
            records.append(_artifact(out, "hemolytik/record.json", "hemolytik", HEMOLYTIK_RECORD, lambda: opener(HEMOLYTIK_RECORD)))
            metadata = json.loads((out / "hemolytik/record.json").read_text())
            files = [f for f in metadata.get("files", []) if str(f.get("key", "")).lower().endswith((".csv", ".xlsx", ".tsv"))]
            archives = [f for f in metadata.get("files", []) if str(f.get("key", "")).lower().endswith(".zip")]
            if len(files) == 1:
                file = files[0]
                url = file["links"].get("content", file["links"].get("self"))
                records.append(_artifact(out, "hemolytik/table" + Path(file["key"]).suffix, "hemolytik", url, lambda: opener(url)))
            elif not files and len(archives) == 1:
                file = archives[0]
                url = file["links"].get("content", file["links"].get("self"))
                records.append(_artifact(out, "hemolytik/archive.zip", "hemolytik", url, lambda: opener(url)))
                with zipfile.ZipFile(out / "hemolytik/archive.zip") as archive:
                    names = [name for name in archive.namelist() if Path(name).name.lower() == "hemolytik2_complete_data.csv"]
                    if len(names) != 1:
                        raise ValueError("Hemolytik archive has no unique Hemolytik2_complete_data.csv; supply --hemolytik table explicitly")
                    content = archive.read(names[0])
                    records.append(_artifact(out, "hemolytik/table.csv", "hemolytik", url + "#" + names[0], lambda: content))
            else:
                raise ValueError("Hemolytik record has no unique supported table/archive; pass --hemolytik with its extracted table")
    manifest = {"schema_version": SCHEMA_VERSION, "created_utc": _utc(), "complete": complete,
                "source_scope": "local full cards" if dbaasp_dir else "DBAASP API index",
                "hemolytik_enabled": include_hemolytik, "release_terms_unresolved": include_hemolytik,
                "artifacts": sorted(records, key=lambda row: row["path"])}
    _immutable(out / "manifest.json", _json_bytes(manifest))
    return manifest


_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNIT = r"(?:[µμu]g\s*/\s*m[lL]|mg\s*/\s*m[lL]|ng\s*/\s*m[lL]|[µμu]g\s*/\s*[lL]|mg\s*/\s*[lL]|[pnµμu]?mol\s*/\s*m[lL]|[pnµμu]?mol\s*/\s*[lL]|[µμu]M|mM|nM|pM|M)"
_MEASURE = rf"(?:[<>≤≥]=?\s*)?{_NUM}(?:\s*(?:[-–—~]|to|±|\+/-)\s*{_NUM})?"


def parse_measurement(text: str | int | float | None) -> dict:
    """Parse a complete numerical expression; never truncate ranges or exponents."""
    raw = "" if text is None else str(text).strip()
    norm = raw.replace("≤", "<=").replace("≥", ">=").replace("−", "-").replace("＋", "+")
    if "," in norm and "." not in norm and re.fullmatch(r"[<>]?=?\s*\d+,\d+", norm):
        norm = norm.replace(",", ".")
    result = {"original_text": raw, "value_lower": None, "value_upper": None,
              "operator": "unknown", "uncertainty": None, "parse_status": "unparsed"}
    matched = re.fullmatch(rf"\s*({_NUM})\s*(?:±|\+/-)\s*({_NUM})\s*", norm)
    if matched:
        value, uncertainty = map(float, matched.groups())
        if not math.isfinite(value) or not math.isfinite(uncertainty):
            return result
        result.update(value_lower=value, value_upper=value, operator="exact", uncertainty=uncertainty, parse_status="parsed")
        return result
    matched = re.fullmatch(rf"\s*({_NUM})\s*(?:[-–—~]|to)\s*({_NUM})\s*", norm)
    if matched:
        lower, upper = map(float, matched.groups())
        if lower > upper or not math.isfinite(lower) or not math.isfinite(upper):
            return result
        result.update(value_lower=lower, value_upper=upper, operator="range", parse_status="parsed")
        return result
    matched = re.fullmatch(rf"\s*([<>]=?|=)?\s*({_NUM})\s*", norm)
    if matched:
        operator, value = matched.groups()
        value = float(value)
        if not math.isfinite(value):
            return result
        operator = operator or "exact"
        if operator == "=":
            operator = "exact"
        result.update(value_lower=None if operator.startswith("<") else value,
                      value_upper=None if operator.startswith(">") else value,
                      operator=operator, parse_status="parsed")
    return result


_MASS = dict(zip("ACDEFGHIKLMNPQRSTVWY", (71.0788, 103.1388, 115.0886, 129.1155,
    147.1766, 57.0519, 137.1411, 113.1594, 128.1741, 113.1594, 131.1926,
    114.1038, 97.1167, 128.1307, 156.1875, 87.0782, 101.1051, 99.1326, 186.2132, 163.1760)))


def peptide_mw(sequence: str) -> float:
    if not sequence or set(sequence) - AA:
        raise ValueError("Molecular weight requires a canonical sequence")
    return sum(_MASS[aa] for aa in sequence) + 18.01528


def concentration_bounds(text, unit: str | None, *, sequence: str = "", chemistry_compatible: bool = False) -> dict:
    result = parse_measurement(text)
    result.update(original_unit=unit, unit="µM", conversion_status="unsupported_unit")
    normalized = (unit or "").replace("μ", "u").replace("µ", "u").replace(" ", "")
    factors = {"M": 1e6, "mM": 1e3, "uM": 1.0, "nM": 1e-3, "pM": 1e-6,
               "mol/l": 1e6, "mmol/l": 1e3, "umol/l": 1., "nmol/l": 1e-3,
               "pmol/l": 1e-6, "mol/ml": 1e9, "umol/ml": 1e3, "nmol/ml": 1., "pmol/ml": 1e-3}
    factor = factors.get(normalized, factors.get(normalized.lower()))
    mass = {"ug/ml": 1000., "mg/ml": 1e6, "ng/ml": 1., "mg/l": 1000., "ug/l": 1.}
    if normalized.lower() in mass:
        if not chemistry_compatible:
            result.update(unit=unit, conversion_status="unknown_molecular_identity")
            return result
        factor = mass[normalized.lower()] / peptide_mw(sequence)
    if factor is None or result["parse_status"] != "parsed":
        result["unit"] = unit
        return result
    for key in ("value_lower", "value_upper", "uncertainty"):
        if result[key] is not None:
            result[key] *= factor
    result["conversion_status"] = "converted"
    return result


def read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read shared/inline/rich strings and sparse cells using standard XML tools."""
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall("m:si", namespace)]
        sheet_names = sorted(n for n in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not sheet_names:
            raise ValueError("XLSX contains no worksheet")
        root = ET.fromstring(archive.read(sheet_names[0]))
    rows = []
    for row in root.findall("m:sheetData/m:row", namespace):
        cells = {}
        for cell in row.findall("m:c", namespace):
            letters = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
            if not letters:
                continue
            index = 0
            for letter in letters.group():
                index = index * 26 + ord(letter) - ord("A") + 1
            value = cell.find("m:v", namespace)
            if cell.attrib.get("t") == "s":
                text = shared[int(value.text)] if value is not None else ""
            elif cell.attrib.get("t") == "inlineStr":
                text = "".join(n.text or "" for n in cell.findall(".//m:t", namespace))
            else:
                text = value.text or "" if value is not None else ""
            cells[index - 1] = text.strip()
        if cells:
            rows.append(cells)
    if not rows:
        return []
    header = rows[0]
    return [{name: row.get(index, "") for index, name in header.items() if name} for row in rows[1:]]


def _name(value) -> str | None:
    if isinstance(value, list) and not value:
        return None
    if isinstance(value, dict):
        value = value.get("name", value.get("value"))
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"na", "n/a", "not available", "unknown", "none", "null", "-"} else None


def _lookup(row: dict, *keys):
    normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in row.items()}
    for key in keys:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", key.lower()))
        if value is not None and value != "":
            return value
    return None


def molecular_identity(sequence: str, raw: dict) -> dict:
    """Strict chemical compatibility. Absence of metadata does not establish L/free/linear."""
    nterm = _name(_lookup(raw, "nTerminus", "N_terminal_modification", "n_terminus", "nterm", "nter"))
    cterm = _name(_lookup(raw, "cTerminus", "C_terminal_modification", "c_terminus", "cterm", "cter"))
    stereo = _name(_lookup(raw, "stereochemistry", "chirality", "aminoAcidConfiguration", "ldmix"))
    linear = _name(_lookup(raw, "linearCyclic", "linear/cyclic", "Linear/Cyclic/Branched", "topology", "lyn_cyc", "structure", "linear"))
    bonds = _lookup(raw, "intrachainBonds", "intraChainBonds", "disulfideBonds")
    if bonds == [] and linear is None:
        linear = "linear"
    if "dAminoAcids" in raw and raw["dAminoAcids"] == [] and stereo is None:
        stereo = "L"
    if any(letter.islower() for letter in sequence):
        stereo = "contains_lowercase_residues_unknown_or_D"
    def key(value):
        return (value or "").lower().replace(" ", "")
    free_n = key(nterm) in {"h", "nh2", "h2n", "free", "unmodified", "freeamine"}
    free_c = key(cterm) in {"oh", "cooh", "free", "unmodified", "freeacid"}
    all_l = key(stereo) in {"l", "all-l", "alll", "l-aminoacids", "l-aminoacid"}
    is_linear = key(linear) in {"linear", "yes", "true"} and not bonds
    extra = _name(_lookup(raw, "other_modifications", "modifications", "unusual_amino_acids", "non_nat"))
    compatible = free_n and free_c and all_l and is_linear and not extra
    return {"n_terminus": nterm, "c_terminus": cterm, "stereochemistry": stereo,
            "topology": linear, "bonds": bonds, "other_modifications": extra,
            "competition_compatible": bool(compatible),
            "chemistry": "canonical_free_linear_L" if compatible else "incompatible_or_unknown"}


def _publication(raw: dict, parent: dict | None = None) -> str | None:
    value = _lookup(raw, "publication_id", "pubmed_id", "pubmed", "pmid", "doi")
    if value is None and parent:
        reference = _name(_lookup(raw, "reference"))
        articles = parent.get("articles") or []
        if reference and reference.isdigit() and 0 < int(reference) <= len(articles):
            # DBAASP uses a one-based article ordinal, not a PubMed identifier.
            article = articles[int(reference) - 1]
            value = _lookup(article, "doi", "pubmed", "pmid")
        elif not reference:
            value = _lookup(parent, "publication_id", "pubmed_id", "pubmed", "pmid", "doi")
    if value is None:
        reference = _name(_lookup(raw, "reference"))
        if reference and re.search(r"10\.\d{4,9}/", reference):
            value = reference
    if isinstance(value, dict):
        value = _lookup(value, "doi", "pubmedId", "pmid")
    value = _name(value)
    if value is None:
        return None
    if "##" in value or re.search(r"\d\s*[;,|]\s*\d", value):
        # A bibliography is not an identified experiment. Keep it in raw data.
        return None
    value = value.strip().lower().replace("https://doi.org/", "").replace("doi:", "")
    value = re.sub(r"^(?:pmid|pubmed)[\s:]*", "", value)
    return "pmid:" + value if value.isdigit() else value


def _endpoint(value: str | None) -> str:
    text = (value or "").strip().upper().replace(" ", "")
    if text == "MINIMUMINHIBITORYCONCENTRATION":
        return "MIC"
    if re.fullmatch(r"(?:MIC|MBC|HC\d+|LC\d+|EC\d+|IC\d+|MHC\d*)", text):
        return text
    if "%" in text:
        return "%lysis"
    return text or "unknown"


def _numeric(value) -> float | None:
    parsed = parse_measurement(_name(value))
    return parsed["value_lower"] if parsed["operator"] == "exact" else None


def _target_identity(target: str | None, explicit_strain: str | None) -> tuple[str | None, str | None]:
    """Normalize species aliases while keeping the full strain context separate."""
    from amp_challenge_2027.data import SPECIES_TO_PANEL

    if not target:
        return None, explicit_strain
    aliases = {k: v for k, v in SPECIES_TO_PANEL.items() if k not in {"Salmonella", "Pseudomonas aeroma"}}
    aliases["Salmonella enterica"] = "S. enterica"
    aliases.update({value: value for value in aliases.values()})
    matches = []
    for alias, name in aliases.items():
        match = re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", target, re.I)
        if match:
            matches.append((name, match))
    if len({name for name, _ in matches}) == 1:
        name, match = max(matches, key=lambda item: len(item[1].group()))
        suffix = target[match.end():].strip(" ,:()") or None
        return name, explicit_strain or suffix
    return target, explicit_strain


def make_observation(molecule: dict, raw: dict, *, source: str, source_id: str,
                     ordinal: int, endpoint: str | None = None, value=None, unit=None,
                     target=None, publication_id=None, dose=None, dose_unit=None) -> dict:
    endpoint = _endpoint(endpoint or _name(_lookup(raw, "endpoint", "activityMeasureGroup", "activityMeasureForLysisGroup", "kind")))
    value = value if value is not None else _lookup(raw, "concentration", "value")
    unit = _name(unit if unit is not None else _lookup(raw, "unit"))
    original_target = _name(target if target is not None else _lookup(raw, "targetSpecies", "targetCell", "target", "organism"))
    target, strain = _target_identity(original_target, _name(_lookup(raw, "strain", "targetStrain")))
    chemistry = molecule["identity"]
    if endpoint == "%lysis":
        bounds = parse_measurement(value)
        bounds.update(unit="%", original_unit=unit, conversion_status="percent")
    else:
        bounds = concentration_bounds(value, unit, sequence=molecule["sequence"], chemistry_compatible=chemistry["competition_compatible"])
    dose_bounds = concentration_bounds(dose, dose_unit, sequence=molecule["sequence"], chemistry_compatible=chemistry["competition_compatible"])
    is_rbc = bool(re.search(r"erythro|red\s*blood|\brbc\b|\bhrbc\b", target or "", re.I))
    if source in {"dramp", "hemolytik"} and re.fullmatch(r"human|sheep|rabbit|horse|mouse|rat|bovine|porcine", target or "", re.I):
        is_rbc = True
    origin = _name(_lookup(raw, "evidence_type", "evidence", "measured_or_predicted")) or "experimental_database_record"
    measured = not re.search(r"predict|in.silico|computational", origin, re.I)
    known_bounds = bounds["parse_status"] == "parsed" and bounds["conversion_status"] in {"converted", "percent"}
    if any(value is not None and value < 0 for value in (bounds["value_lower"], bounds["value_upper"])):
        known_bounds = False
    if endpoint == "%lysis" and any(value is not None and value > 100 for value in (bounds["value_lower"], bounds["value_upper"])):
        known_bounds = False
    # Percent-lysis supervision is uninterpretable without its concentration.
    dose_valid = endpoint != "%lysis" or dose_bounds["conversion_status"] == "converted"
    eligible = bool(chemistry["competition_compatible"] and measured and known_bounds and dose_valid and target)
    if endpoint in {"unknown", "MHC", "IC50", "EC50"}:
        # Preserve ambiguous endpoints but never promote them to HC50 supervision.
        eligible = False
    token = {"endpoint": endpoint, "target": target, "strain": strain,
             "rbc_species": _name(_lookup(raw, "rbc_species", "red_cell_species")) or (target if is_rbc else None),
             "medium": _name(_lookup(raw, "medium", "cultureMedium")), "chemistry": chemistry["chemistry"],
             "operator": bounds["operator"], "unit": bounds["unit"],
             "value_lower": bounds["value_lower"], "value_upper": bounds["value_upper"],
             "uncertainty": bounds["uncertainty"],
             "dose_lower": dose_bounds["value_lower"] if dose_bounds["conversion_status"] == "converted" else None,
             "dose_upper": dose_bounds["value_upper"] if dose_bounds["conversion_status"] == "converted" else None,
             "dose_operator": dose_bounds["operator"], "dose_unit": "µM" if dose is not None else None,
             "ph": _numeric(_lookup(raw, "ph")), "temperature": _numeric(_lookup(raw, "temperature")),
             "assay": _name(_lookup(raw, "assay")), "method": _name(_lookup(raw, "method")),
             "incubation_time": _name(_lookup(raw, "incubation_time", "incubation")),
             "rbc_concentration": _name(_lookup(raw, "rbc_concentration")),
             "ionic_strength": _name(_lookup(raw, "ionic_strength", "ionicStrength")),
             "salt_type": _name(_lookup(raw, "salt_type", "saltType")),
             "publication_id": publication_id or _publication(raw), "original_text": bounds["original_text"]}
    context = {k: _lookup(raw, k) for k in ("note", "assay", "method", "incubation", "incubation_time", "rbc_concentration",
                                           "cfu", "ionicStrength", "saltType")}
    return {"molecule_id": molecule["molecule_id"], "sequence": molecule["sequence"],
            "source": source, "source_id": source_id, "ordinal": ordinal,
            "evidence_type": origin, "condition": token, "assay_context": context,
            "raw": raw, "original_target": original_target, "original_unit": unit, "conversion_status": bounds["conversion_status"],
            "supervision_eligible": eligible, "sources": [{"source": source, "source_id": source_id, "ordinal": ordinal}]}


def _molecule(sequence: str, raw: dict, source: str, source_id: str) -> dict | None:
    sequence = re.sub(r"\s+", "", sequence or "")
    normalized = sequence.upper()
    if not 8 <= len(normalized) <= 50 or set(normalized) - AA:
        return None
    identity = molecular_identity(sequence, raw)
    identity_key = identity["chemistry"] if identity["competition_compatible"] else identity
    identifier = _sha(_json_bytes({"sequence": normalized, "identity": identity_key}))[:24]
    return {"molecule_id": identifier, "sequence": normalized, "identity": identity,
            "sources": [{"source": source, "source_id": source_id}]}


def import_dbaasp_card(raw: dict) -> tuple[list[dict], list[dict]]:
    monomers = raw.get("monomers") or []
    if len(monomers) > 1:
        return [], []
    seq = raw.get("sequence") or (monomers[0].get("sequence") if monomers else "")
    info = {**(monomers[0] if monomers else {}), **raw}
    identifier = str(raw.get("id", "unknown"))
    molecule = _molecule(seq, info, "dbaasp", identifier)
    if molecule is None:
        return [], []
    observations = []
    for ordinal, record in enumerate(raw.get("targetActivities") or []):
        observations.append(make_observation(molecule, record, source="dbaasp", source_id=identifier,
                                             ordinal=ordinal, publication_id=_publication(record, raw)))
    offset = len(observations)
    for ordinal, record in enumerate(raw.get("hemoliticCytotoxicActivities") or []):
        kind = _name(record.get("activityMeasureForLysisGroup")) or "unknown"
        if "%" in kind:
            percent_match = re.fullmatch(rf"\s*({_MEASURE})\s*%\s*(?:ha?emolysis|lysis)?\s*", kind, re.I)
            percent = percent_match.group(1) if percent_match else kind
            observations.append(make_observation(molecule, record, source="dbaasp", source_id=identifier,
                ordinal=offset + ordinal, endpoint="%lysis", value=percent, unit="%",
                dose=record.get("concentration"), dose_unit=_name(record.get("unit")), publication_id=_publication(record, raw)))
        else:
            observations.append(make_observation(molecule, record, source="dbaasp", source_id=identifier,
                ordinal=offset + ordinal, endpoint=kind, publication_id=_publication(record, raw)))
    return [molecule], observations


def _text_assays(text: str, *, hemolysis: bool = False) -> list[dict]:
    """Conservative clause parser; ambiguous free text remains in raw evidence."""
    rows = []
    for clause in re.split(r"[;\n]", text or ""):
        previous_end = 0
        for matched in re.finditer(rf"\b(MIC|MBC|HC\d+|LC\d+|MHC\d*|IC50|EC50)\s*(?:[:=]\s*)?({_MEASURE})\s*({_UNIT})\b", clause, re.I):
            endpoint, value, unit = matched.groups()
            target = clause[previous_end:matched.start()].strip(" ,:()") or None
            previous_end = matched.end()
            rows.append({"endpoint": endpoint, "concentration": value, "unit": unit,
                         "target": target, "original_clause": clause})
        if hemolysis:
            matched = re.search(rf"({_MEASURE})\s*%\s*(?:ha?emolysis|lysis)?[^;]*?\b(?:at|@)\s*({_MEASURE})\s*({_UNIT})", clause, re.I)
            if matched:
                value, dose, unit = matched.groups()
                target_match = re.search(r"\b(human|sheep|rabbit|horse|mouse|rat|bovine|porcine)\b", clause, re.I)
                rows.append({"endpoint": "%lysis", "value": value, "unit": "%", "dose": dose, "dose_unit": unit,
                             "target": target_match.group().lower() if target_match else None,
                             "original_clause": clause})
    return rows


def import_dramp_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    molecules, observations = [], []
    for index, raw in enumerate(rows):
        identifier = str(_lookup(raw, "DRAMP_ID", "id") or index)
        molecule = _molecule(str(_lookup(raw, "sequence") or ""), raw, "dramp", identifier)
        if molecule is None:
            continue
        molecules.append(molecule)
        # Activity is a category, not a numerical assay column.
        texts = [(str(_lookup(raw, "Target_Organism") or ""), False),
                 (str(_lookup(raw, "Hemolytic_activity") or ""), True)]
        ordinal = 0
        for text, hemo in texts:
            assays = _text_assays(text, hemolysis=hemo)
            if text and not assays:
                assays = [{"endpoint": "unknown", "value": None, "original_clause": text}]
            for assay in assays:
                observations.append(make_observation(molecule, {**raw, **assay}, source="dramp", source_id=identifier,
                    ordinal=ordinal, endpoint=assay["endpoint"], value=assay.get("value", assay.get("concentration")),
                    unit=assay.get("unit"), target=assay.get("target"), dose=assay.get("dose"),
                    dose_unit=assay.get("dose_unit"), publication_id=_publication(raw)))
                ordinal += 1
    return molecules, observations


def import_hemolytik_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Explicit optional adapter. Retains source terms flag in snapshot/dataset."""
    molecules, observations = [], []
    for index, raw in enumerate(rows):
        identifier = str(_lookup(raw, "id", "hemolytik_id") or index)
        molecule = _molecule(str(_lookup(raw, "sequence", "peptide_sequence", "seq") or ""), raw, "hemolytik", identifier)
        if molecule is None:
            continue
        molecules.append(molecule)
        # Official CSV stores free-text activity (e.g. 'LC50 = 1.4±0.2 µM'),
        # seq/cter/nter/lyn_cyc/ldmix/non_nat chemistry and an RBC 'source'.
        assays = _text_assays(str(_lookup(raw, "activity") or ""), hemolysis=True)
        if not assays:
            assays = [{"endpoint": _name(_lookup(raw, "endpoint", "measure")) or "unknown",
                       "value": _lookup(raw, "value", "concentration", "hemolysis"),
                       "unit": _lookup(raw, "unit", "units"), "dose": _lookup(raw, "dose", "peptide_concentration"),
                       "dose_unit": _lookup(raw, "dose_unit")}]
        for ordinal, assay in enumerate(assays):
            observations.append(make_observation(molecule, raw, source="hemolytik", source_id=identifier, ordinal=ordinal,
                endpoint=assay["endpoint"], value=assay.get("value", assay.get("concentration")), unit=assay.get("unit"),
                target=_lookup(raw, "rbc_species", "rbc_source", "target", "red_cell_source", "source"),
                dose=assay.get("dose"), dose_unit=assay.get("dose_unit")))
    return molecules, observations


def deduplicate_observations(observations: list[dict]) -> list[dict]:
    """Merge copied measurements only with a shared identified study and context."""
    retained = {}
    for row in observations:
        condition = row["condition"]
        provenance = condition["publication_id"] or [row["source"], row["source_id"], row["ordinal"]]
        comparable = {k: v for k, v in condition.items() if k != "original_text"}
        key = _sha(_json_bytes({"molecule": row["molecule_id"], "study": provenance,
                               "condition": comparable, "context": row["assay_context"],
                               "evidence_type": row["evidence_type"], "eligible": row["supervision_eligible"]}))
        if key in retained and any(s["source"] == row["source"] and s != row["sources"][0]
                                   for s in retained[key]["sources"]):
            # Distinct records in one source may be experimental replicates.
            # Cross-source copies can merge; do not collapse within-source repeats.
            key = _sha(_json_bytes({"measurement_key": key, "occurrence": row["sources"][0]}))
        if key in retained:
            retained[key]["sources"].extend(row["sources"])
        else:
            retained[key] = {**row, "observation_id": key[:24], "sources": list(row["sources"])}
    return sorted(retained.values(), key=lambda row: row["observation_id"])


def _table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        return read_xlsx_rows(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))


def _replay_molecules(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith(">"):
        records, seq = [], []
        for line in text.splitlines():
            if line.startswith(">"):
                if seq:
                    records.append({"sequence": "".join(seq)})
                seq = []
            else:
                seq.append(line.strip())
        if seq:
            records.append({"sequence": "".join(seq)})
    else:
        records = list(csv.DictReader(io.StringIO(text)))
        if records and _lookup(records[0], "sequence", "seq") is None:
            raise ValueError("MarLys CSV must have a sequence or seq column")
    result = []
    for i, row in enumerate(records):
        molecule = _molecule(str(_lookup(row, "sequence", "seq") or ""), {}, "marlys", str(i))
        if molecule:
            result.append(molecule)
    return result


def prepare_dataset(snapshot: Path, out: Path, *, seed: int = 2027, threshold: float = .8,
                    allow_incomplete: bool = False) -> dict:
    """Prepare immutable observation tables and a full-union sequence-family split."""
    from amp_challenge_2027.generalization import assign_families, family_components

    snapshot, out = Path(snapshot), Path(out)
    manifest = verify_snapshot(snapshot)
    if not manifest["complete"] and not allow_incomplete:
        raise ValueError("Snapshot is explicitly incomplete; --allow-incomplete is required for smoke tests")
    recipe = {"schema_version": SCHEMA_VERSION, "snapshot_sha256": _sha((snapshot / "manifest.json").read_bytes()),
              "parser_sha256": _sha(Path(__file__).read_bytes()),
              "seed": seed, "threshold": threshold, "allow_incomplete": allow_incomplete}
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        if previous["recipe"] != recipe:
            raise ValueError("Prepared dataset inputs differ; use a new output directory")
        for rel, digest in previous["outputs"].items():
            if _sha((out / rel).read_bytes()) != digest:
                raise ValueError(f"Prepared dataset hash mismatch: {out / rel}")
        return json.loads((out / "dataset.json").read_text())
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prepared dataset directory is not empty; use a new output directory")
    molecules, observations = [], []
    for artifact in manifest["artifacts"]:
        path = snapshot / artifact["path"]
        source = artifact["source"]
        if source == "marlys":
            molecules.extend(_replay_molecules(path))
        elif source == "dbaasp" and "/cards/" in artifact["path"]:
            mols, obs = import_dbaasp_card(json.loads(path.read_text()))
            molecules.extend(mols)
            observations.extend(obs)
        elif source == "dramp" and path.suffix.lower() == ".xlsx":
            mols, obs = import_dramp_rows(read_xlsx_rows(path))
            molecules.extend(mols)
            observations.extend(obs)
        elif source == "hemolytik" and path.suffix.lower() in {".xlsx", ".csv", ".tsv"}:
            mols, obs = import_hemolytik_rows(_table(path))
            molecules.extend(mols)
            observations.extend(obs)
    unique = {}
    for row in molecules:
        if row["molecule_id"] in unique:
            unique[row["molecule_id"]]["sources"].extend(row["sources"])
        else:
            unique[row["molecule_id"]] = row
    molecules = sorted(unique.values(), key=lambda row: row["molecule_id"])
    observations = deduplicate_observations(observations)
    sequences = sorted({row["sequence"] for row in molecules})
    if not sequences:
        raise ValueError("No valid canonical 8–50-residue sequences in snapshot")
    print(f"[conditional-data] family graph over {len(sequences)} unique sequences including unlabeled bridges", flush=True)
    families = family_components(sequences, threshold, progress=lambda msg: print("[conditional-data] " + msg, flush=True))
    splits = assign_families(families, seed)
    by_sequence = {seq: [] for seq in sequences}
    for row in observations:
        row["split"], row["family"] = splits[row["sequence"]], families[row["sequence"]]
        if row["supervision_eligible"]:
            by_sequence[row["sequence"]].append(row["condition"])
    for row in molecules:
        row["split"], row["family"] = splits[row["sequence"]], families[row["sequence"]]
    examples = [{"sequence": seq, "split": splits[seq], "family": families[seq],
                 "conditions": by_sequence[seq], "supervision_eligible": bool(by_sequence[seq])} for seq in sequences]
    support = {"sequences": len(sequences), "molecules": len(molecules), "observations": len(observations),
               "supervision_sequences": sum(e["supervision_eligible"] for e in examples),
               "splits": dict(Counter(splits.values())),
               "annotated_splits": dict(Counter(e["split"] for e in examples if e["conditions"])),
               "eligible_endpoints": dict(Counter(o["condition"]["endpoint"] for o in observations if o["supervision_eligible"])),
               "conversion_status": dict(Counter(o["conversion_status"] for o in observations)),
               "incompatible_or_unknown_molecules": sum(not m["identity"]["competition_compatible"] for m in molecules),
               "unparsed_or_ineligible_observations": sum(not o["supervision_eligible"] for o in observations),
               "hemolytik_release_terms_unresolved": manifest.get("release_terms_unresolved", False),
               "split_metric": "Levenshtein.ratio normalized indel similarity", "family_boundary": threshold,
               "fractions": {"train": .6, "validation": .15, "calibration": .1, "test": .15},
               "warm_start_caveat": "Incumbent pretraining may include held-out sequences; adaptation is not clean generalization evidence."}
    dataset = {"schema_version": SCHEMA_VERSION, "recipe": recipe, "examples": examples,
               "support": support, "source_manifest": manifest}
    outputs = {"dataset.json": _json_bytes(dataset), "support.json": _json_bytes(support),
               "molecules.jsonl": b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n" for row in molecules),
               "observations.jsonl": b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n" for row in observations)}
    for name, data in outputs.items():
        _immutable(out / name, data)
    _immutable(out / "manifest.json", _json_bytes({"recipe": recipe, "outputs": {k: _sha(v) for k, v in outputs.items()}}))
    return dataset
