import math

import pytest

from vlm_comparison.metrics import (
    ExactMatchAccuracy,
    TokenF1,
    metric_from_name,
    normalize_text,
)


def test_normalize_text_collapses_whitespace_and_lowercases():
    assert normalize_text("  Two   CATS ") == "two cats"


def test_exact_match_manual_value():
    preds = ["red", "GREEN", "blue", "purple"]
    refs = ["red", "green", "blue", "orange"]
    # Three of four match after normalization -> 0.75 exactly.
    score = ExactMatchAccuracy().score(preds, refs)
    assert score == pytest.approx(0.75)


def test_exact_match_all_correct_is_one():
    preds = ["a", "b", "c"]
    assert ExactMatchAccuracy().score(preds, preds) == pytest.approx(1.0)


def test_exact_match_all_wrong_is_zero():
    assert ExactMatchAccuracy().score(["x", "y"], ["a", "b"]) == pytest.approx(0.0)


def test_token_f1_partial_overlap_manual():
    # prediction "two big cats", reference "two cats"
    # shared tokens = {two, cats} -> shared = 2
    # precision = 2/3, recall = 2/2 = 1.0
    # f1 = 2 * (2/3 * 1) / (2/3 + 1) = (4/3) / (5/3) = 0.8
    f1 = TokenF1()._pair_f1("two big cats", "two cats")
    assert f1 == pytest.approx(0.8)


def test_token_f1_no_overlap_is_zero():
    assert TokenF1()._pair_f1("dog", "cat") == pytest.approx(0.0)


def test_token_f1_identical_is_one():
    assert TokenF1()._pair_f1("at the beach", "at the beach") == pytest.approx(1.0)


def test_token_f1_mean_over_pairs():
    preds = ["two cats", "wrong"]
    refs = ["two cats", "right"]
    # First pair f1 = 1.0, second pair f1 = 0.0 -> mean 0.5.
    assert TokenF1().score(preds, refs) == pytest.approx(0.5)


def test_token_f1_respects_multiplicity():
    # prediction has "cat" twice, reference once. shared counts once.
    # precision = 1/2, recall = 1/1 -> f1 = 2*(0.5*1)/(0.5+1) = 1/1.5 = 0.6667
    f1 = TokenF1()._pair_f1("cat cat", "cat")
    assert f1 == pytest.approx(2 / 3)


def test_metric_length_mismatch_raises():
    with pytest.raises(ValueError):
        ExactMatchAccuracy().score(["a"], ["a", "b"])


def test_metric_empty_raises():
    with pytest.raises(ValueError):
        ExactMatchAccuracy().score([], [])


def test_metric_from_name_roundtrip():
    assert isinstance(metric_from_name("exact_match"), ExactMatchAccuracy)
    assert isinstance(metric_from_name("token_f1"), TokenF1)


def test_metric_from_name_unknown_raises():
    with pytest.raises(KeyError):
        metric_from_name("bleu")
