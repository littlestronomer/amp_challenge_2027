"""Measured assay conditions for the research decoder.

Schema fitting is deliberately independent of torch and only consumes the examples
passed by the caller (the training partition). Missing values are not biological
negatives. Publication identifiers and original text stay in provenance records;
they are not predictive features or generation targets.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

CATEGORICAL_FIELDS = (
    "endpoint", "target", "strain", "rbc_species", "medium", "chemistry",
    "operator", "dose_operator", "unit", "dose_unit",
    "assay", "method", "incubation_time", "rbc_concentration", "ionic_strength", "salt_type",
)
NUMERIC_FIELDS = (
    "value_lower", "value_upper", "dose_lower", "dose_upper", "ph",
    "temperature", "uncertainty",
)
SCHEMA_VERSION = 1


def _category(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Condition values must be finite; use None for missing bounds")
    return result


def _transform(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def fit_condition_schema(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit categorical vocabularies and numeric normalization on train examples.

    If split labels are present, reject any non-training example. The vocabulary
    reserves 0 for missing and 1 for a category unseen during fitting. Numeric
    normalization uses signed log1p then training-set mean and standard deviation.
    """
    tokens: list[Mapping[str, Any]] = []
    for example in examples:
        if example.get("split") not in (None, "train", "training"):
            raise ValueError("Condition schemas may only be fitted on the training split")
        tokens.extend(example.get("conditions") or [])
    vocabularies = {}
    for field in CATEGORICAL_FIELDS:
        categories = sorted({v for token in tokens if (v := _category(token.get(field))) is not None})
        vocabularies[field] = {value: i + 2 for i, value in enumerate(categories)}
    normalizers = {}
    for field in NUMERIC_FIELDS:
        values = [_transform(value) for token in tokens if (value := _number(token.get(field))) is not None]
        mean = math.fsum(values) / len(values) if values else 0.0
        variance = math.fsum((x - mean) ** 2 for x in values) / len(values) if values else 0.0
        normalizers[field] = {
            "mean": mean, "scale": max(math.sqrt(variance), 1e-6) if variance > 0 else 1.0,
            "count": len(values), "transform": "signed_log1p",
            "min": min(values) if values else None, "max": max(values) if values else None,
        }
    return {
        "version": SCHEMA_VERSION,
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "numeric_fields": list(NUMERIC_FIELDS),
        "vocabularies": vocabularies,
        "normalizers": normalizers,
        "null_token": True,
    }


def validate_condition_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("version") != SCHEMA_VERSION:
        raise ValueError("Unsupported assay condition schema version")
    if tuple(schema.get("categorical_fields", ())) != CATEGORICAL_FIELDS:
        raise ValueError("Assay categorical fields do not match schema version 1")
    if tuple(schema.get("numeric_fields", ())) != NUMERIC_FIELDS:
        raise ValueError("Assay numerical fields do not match schema version 1")
    for field in CATEGORICAL_FIELDS:
        vocabulary = schema["vocabularies"][field]
        if sorted(vocabulary.values()) != list(range(2, len(vocabulary) + 2)):
            raise ValueError(f"Invalid vocabulary indices for {field}")
    for field in NUMERIC_FIELDS:
        normalizer = schema["normalizers"][field]
        if normalizer.get("transform") != "signed_log1p":
            raise ValueError(f"Unknown numeric transform for {field}")
        if not math.isfinite(normalizer["mean"]) or not math.isfinite(normalizer["scale"]) or normalizer["scale"] <= 0:
            raise ValueError(f"Invalid numeric normalization for {field}")


def condition_support(conditions: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe extrapolation without claiming that supported requests are safe."""
    validate_condition_schema(schema)
    warnings = []
    for index, token in enumerate(conditions):
        for field in CATEGORICAL_FIELDS:
            value = _category(token.get(field))
            if value is not None and value not in schema["vocabularies"][field]:
                warnings.append({"token": index, "field": field, "reason": "unseen_category", "value": value})
        for field in NUMERIC_FIELDS:
            value = _number(token.get(field))
            if value is None:
                continue
            norm = schema["normalizers"][field]
            transformed = _transform(value)
            if norm["count"] == 0 or not norm["min"] <= transformed <= norm["max"]:
                warnings.append({"token": index, "field": field, "reason": "outside_training_range", "value": value})
    return warnings


def collate_assay_conditions(
    batches: Sequence[Sequence[Mapping[str, Any]]], schema: Mapping[str, Any], device=None,
    dropout: float = 0.0, generator=None,
) -> dict[str, Any]:
    """Encode a batch of observation lists with an always-visible NULL token.

    Returned tensors are categorical (B,M,C), values/observed (B,M,N), and
    valid_mask (B,M). Token dropout only applies to measured tokens, never NULL.
    CPU or CUDA torch generators are supported independently of output device.
    """
    import torch

    validate_condition_schema(schema)
    if not batches:
        raise ValueError("Cannot collate an empty condition batch")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("Condition dropout must be in [0, 1]")
    batch_size, width = len(batches), 1 + max(map(len, batches))
    categorical = torch.zeros((batch_size, width, len(CATEGORICAL_FIELDS)), dtype=torch.long)
    values = torch.zeros((batch_size, width, len(NUMERIC_FIELDS)), dtype=torch.float32)
    observed = torch.zeros_like(values, dtype=torch.bool)
    valid_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    valid_mask[:, 0] = True
    for row, tokens in enumerate(batches):
        for column, token in enumerate(tokens, start=1):
            # Empty annotations carry no information and are not measured tokens.
            has_data = False
            for feature, field in enumerate(CATEGORICAL_FIELDS):
                value = _category(token.get(field))
                if value is not None:
                    has_data = True
                    categorical[row, column, feature] = schema["vocabularies"][field].get(value, 1)
            for feature, field in enumerate(NUMERIC_FIELDS):
                value = _number(token.get(field))
                if value is not None:
                    has_data = True
                    normalizer = schema["normalizers"][field]
                    values[row, column, feature] = (_transform(value) - normalizer["mean"]) / normalizer["scale"]
                    observed[row, column, feature] = True
            valid_mask[row, column] = has_data
    if dropout:
        rng_device = generator.device if generator is not None else "cpu"
        keep = torch.rand(valid_mask.shape, generator=generator, device=rng_device).cpu() >= dropout
        keep[:, 0] = True
        valid_mask &= keep
    # Physically clear dropped values in addition to the attention mask.
    categorical.masked_fill_(~valid_mask[..., None], 0)
    observed &= valid_mask[..., None]
    values.masked_fill_(~observed, 0)
    return {name: value.to(device) for name, value in {
        "categorical": categorical, "values": values, "observed": observed,
        "valid_mask": valid_mask,
    }.items()}
