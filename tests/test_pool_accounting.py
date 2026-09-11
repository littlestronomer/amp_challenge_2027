import pytest

pd = pytest.importorskip("pandas")

from audit_generator_pools import accounting, diagnose  # noqa: E402


def test_exclusive_accounting_and_prefixes():
    frame = pd.DataFrame({"sequence": ["AAAAAAAA", "AAAAAAAA", "CCCCCCCC", "CCCCCCCC", "XX", "DDDDDDDD"],
                          "activity": [.9]*6, "risk": [.1]*6, "evaluation_risk": [.2]*6})
    counts, categories = accounting(frame, {"AAAAAAAA"})
    assert counts["exact_reference"] == 2
    assert counts["reference_unique"] == 1
    assert counts["repeated_nonreference"] == 1
    assert counts["invalid"] == 1
    assert counts["unique_nonreference"] == 2
    assert len(categories) == sum(counts[k] for k in
        ("invalid", "exact_reference", "repeated_nonreference", "unique_nonreference"))
    totals, strata, families = diagnose(frame, {"AAAAAAAA"}, [3, 6])
    assert totals[0]["qualifying_unique"] == 1
    assert totals[1]["qualifying_unique"] == 2
    assert totals[1]["qualifying_yield_per_1000"] == pytest.approx(1000/3)
    assert sum(r["size"] for r in families if r["draws"] == 6) == 2
    assert sum(r["qualifying_unique"] for r in strata if r["draws"] == 6) == 2


def test_family_grouping_and_bad_scores():
    frame = pd.DataFrame({"sequence": ["AAAAAAAA", "AAAAAAAC", "CCCCCCCC"],
                          "activity": [.9]*3, "risk": [.1]*3, "evaluation_risk": [.2]*3})
    totals, _, families = diagnose(frame, set(), [3])
    assert totals[0]["qualifying_greedy_families"] == 2
    assert sorted(r["size"] for r in families) == [1, 2]
    frame.loc[0, "activity"] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        diagnose(frame, set(), [3])


def test_invalid_scores_allowed_only_for_invalid_sequences():
    frame = pd.DataFrame({"sequence": [""], "activity": [float("nan")],
                          "risk": [float("nan")], "evaluation_risk": [float("nan")]})
    totals, _, families = diagnose(frame, set(), [1])
    assert totals[0]["invalid"] == 1
    assert not families
    with pytest.raises(ValueError, match="budget"):
        diagnose(frame, set(), [2])
