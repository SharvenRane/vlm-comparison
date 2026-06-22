import pytest

from vlm_comparison.adapters import (
    ConstantAdapter,
    KeywordMatchAdapter,
    NearestExampleAdapter,
)
from vlm_comparison.benchmark import build_default_benchmark
from vlm_comparison.harness import run_comparison
from vlm_comparison.metrics import ExactMatchAccuracy, TokenF1


def test_harness_runs_and_ranks_three_adapters():
    bench = build_default_benchmark()
    color_support = [item for item in bench if item.category == "color"]

    adapters = [
        KeywordMatchAdapter(),
        NearestExampleAdapter(color_support, name="nearest-color"),
        ConstantAdapter("red", name="always-red"),
    ]
    report = run_comparison(adapters, bench, ExactMatchAccuracy())

    # Every adapter appears exactly once.
    assert set(report.rankings) == {"keyword-match", "nearest-color", "always-red"}
    assert len(report.results) == 3

    # The keyword adapter solves the benchmark, so it must rank first.
    assert report.best.adapter_name == "keyword-match"
    assert report.best.score == pytest.approx(1.0)


def test_ranking_is_sorted_best_to_worst():
    bench = build_default_benchmark()
    adapters = [
        ConstantAdapter("red", name="always-red"),
        KeywordMatchAdapter(),
    ]
    report = run_comparison(adapters, bench, ExactMatchAccuracy())
    scores = [r.score for r in report.results]
    assert scores == sorted(scores, reverse=True)


def test_score_matches_manual_computation():
    """The harness score must equal a metric computed by hand on the predictions."""

    bench = build_default_benchmark()
    metric = ExactMatchAccuracy()
    adapter = ConstantAdapter("red", name="always-red")

    report = run_comparison([adapter], bench, metric)
    result = report.results[0]

    # Manual computation: count items whose reference answer is exactly "red".
    references = bench.answers
    manual_hits = sum(ref == "red" for ref in references)
    manual_score = manual_hits / len(references)

    assert result.score == pytest.approx(manual_score)

    # Also confirm against the metric applied directly to the stored predictions.
    pred_texts = [p.text for p in result.predictions]
    direct = metric.score(pred_texts, references)
    assert result.score == pytest.approx(direct)


def test_constant_adapter_score_is_red_fraction():
    bench = build_default_benchmark()
    report = run_comparison(
        [ConstantAdapter("red", name="always-red")], bench, ExactMatchAccuracy()
    )
    # The default benchmark has exactly two "red" answers out of ten items.
    assert report.score_for("always-red") == pytest.approx(2 / 10)


def test_token_f1_metric_runs_through_harness():
    bench = build_default_benchmark()
    report = run_comparison([KeywordMatchAdapter()], bench, TokenF1())
    assert report.metric_name == "token_f1"
    # Keyword adapter solves the benchmark, so token F1 is also perfect.
    assert report.best.score == pytest.approx(1.0)


def test_as_table_has_one_row_per_adapter_with_ranks():
    bench = build_default_benchmark()
    adapters = [KeywordMatchAdapter(), ConstantAdapter("red", name="always-red")]
    report = run_comparison(adapters, bench, ExactMatchAccuracy())
    table = report.as_table()
    assert [row["rank"] for row in table] == [1, 2]
    assert table[0]["adapter"] == "keyword-match"


def test_duplicate_adapter_names_raise():
    bench = build_default_benchmark()
    with pytest.raises(ValueError):
        run_comparison(
            [ConstantAdapter("a", name="dup"), ConstantAdapter("b", name="dup")],
            bench,
            ExactMatchAccuracy(),
        )


def test_empty_adapter_list_raises():
    bench = build_default_benchmark()
    with pytest.raises(ValueError):
        run_comparison([], bench, ExactMatchAccuracy())


def test_tie_breaking_keeps_input_order():
    bench = build_default_benchmark()
    # Two identical constant adapters with different names tie on score.
    adapters = [
        ConstantAdapter("zzz", name="first"),
        ConstantAdapter("zzz", name="second"),
    ]
    report = run_comparison(adapters, bench, ExactMatchAccuracy())
    assert report.rankings == ["first", "second"]
