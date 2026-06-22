"""The comparison harness.

The harness runs each adapter over the benchmark, gathers predictions, scores
them with a metric, and ranks the adapters from best to worst. It also keeps the
raw predictions so a caller can inspect per item behavior or compute a second
metric without re running the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .adapters import ModelAdapter, Prediction
from .benchmark import Benchmark
from .metrics import Metric


@dataclass
class AdapterResult:
    """The outcome of running one adapter over the benchmark.

    Attributes:
        adapter_name: name of the adapter.
        metric_name: name of the metric used for the score.
        score: the aggregate metric value.
        predictions: the raw per item predictions in benchmark order.
    """

    adapter_name: str
    metric_name: str
    score: float
    predictions: List[Prediction]


@dataclass
class ComparisonReport:
    """A ranked comparison across adapters.

    The results list is sorted from best to worst according to the metric's
    direction. The rankings list gives adapter names in that same order.
    """

    metric_name: str
    results: List[AdapterResult]

    @property
    def rankings(self) -> List[str]:
        return [result.adapter_name for result in self.results]

    @property
    def best(self) -> AdapterResult:
        if not self.results:
            raise ValueError("report has no results")
        return self.results[0]

    def score_for(self, adapter_name: str) -> float:
        for result in self.results:
            if result.adapter_name == adapter_name:
                return result.score
        raise KeyError(f"no result for adapter '{adapter_name}'")

    def as_table(self) -> List[Dict[str, object]]:
        """Return a plain list of row dicts for printing or serialization."""

        rows = []
        for rank, result in enumerate(self.results, start=1):
            rows.append(
                {
                    "rank": rank,
                    "adapter": result.adapter_name,
                    "metric": result.metric_name,
                    "score": result.score,
                }
            )
        return rows


def run_comparison(
    adapters: Sequence[ModelAdapter],
    benchmark: Benchmark,
    metric: Metric,
) -> ComparisonReport:
    """Run every adapter over the benchmark and rank them by the metric.

    Args:
        adapters: the models to compare. Names must be unique.
        benchmark: the evaluation set.
        metric: the scoring metric.

    Returns:
        A :class:`ComparisonReport` sorted from best to worst. Ties keep the
        original adapter order, which makes the ranking deterministic.
    """

    if len(adapters) == 0:
        raise ValueError("need at least one adapter to compare")
    if len(benchmark) == 0:
        raise ValueError("benchmark must contain at least one item")

    names = [adapter.name for adapter in adapters]
    if len(set(names)) != len(names):
        raise ValueError(f"adapter names must be unique, got {names}")

    items = list(benchmark)
    references = benchmark.answers

    results: List[AdapterResult] = []
    for adapter in adapters:
        predictions = adapter.predict(items)
        if len(predictions) != len(items):
            raise ValueError(
                f"adapter '{adapter.name}' returned {len(predictions)} "
                f"predictions for {len(items)} items"
            )
        # Align predictions to benchmark order by item id to be safe.
        by_id = {p.item_id: p.text for p in predictions}
        ordered_texts = []
        for item in items:
            if item.item_id not in by_id:
                raise ValueError(
                    f"adapter '{adapter.name}' did not predict for item "
                    f"'{item.item_id}'"
                )
            ordered_texts.append(by_id[item.item_id])
        score = metric.score(ordered_texts, references)
        results.append(
            AdapterResult(
                adapter_name=adapter.name,
                metric_name=metric.name,
                score=score,
                predictions=predictions,
            )
        )

    # Sort best to worst. Python's sort is stable, so equal scores keep input
    # order. We negate when higher is better so the best lands first.
    sign = -1.0 if metric.higher_is_better else 1.0
    results_sorted = sorted(results, key=lambda r: sign * r.score)

    return ComparisonReport(metric_name=metric.name, results=results_sorted)
