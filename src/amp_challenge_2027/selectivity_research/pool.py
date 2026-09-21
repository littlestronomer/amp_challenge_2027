from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import MDR_PANEL_GENERA
from amp_challenge_2027.data import iter_fasta
from amp_challenge_2027.props import is_plausible
from .contracts import validate_pool_frame


def digest_sequences(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode())
        h.update(b"\0")
    return h.hexdigest()


def read_pool(path: Path, genera: list[str]):
    import pandas as pd
    frame = pd.read_csv(path, keep_default_na=False)
    validate_pool_frame(frame, genera=genera)
    return frame


def write_pool(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_rows(cell: dict, risk: np.ndarray, genera: list[str]) -> list[dict]:
    sequences = cell["sequences"]
    if len(risk) != len(sequences):
        raise ValueError("Risk array is not aligned with source library")
    top_rank = {sequence: rank for rank, sequence in enumerate(cell["top"], 1)}
    panel = np.asarray(cell["scores"]["panel"], dtype=float)
    rows = []
    for index, sequence in enumerate(sequences):
        row = {
            "sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "library_index": index, "source_seed": cell["seed"],
            "activity": float(cell["scores"]["activity"][index]),
            "hemolysis_risk": float(risk[index]),
            "conformity": float(cell["scores"]["conformity"][index]),
            "precision": float(cell["scores"]["precision"][index]),
            "plausible": bool(is_plausible(sequence)),
            "incumbent_member": sequence in top_rank,
            "incumbent_rank": top_rank.get(sequence, ""),
        }
        row.update({f"panel:{genus}": float(panel[index, j]) for j, genus in enumerate(genera)})
        rows.append(row)
    return rows


def numeric_array(frame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=np.float64)
