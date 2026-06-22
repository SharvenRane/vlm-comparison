"""Metrics for scoring model predictions against reference answers.

Two metrics ship here. Exact match accuracy is the fraction of predictions that
equal the reference after light normalization. Token F1 measures word level
overlap and rewards partially correct answers. Both operate on plain strings so
their values are easy to check by hand.
"""

from __future__ import annotations

from typing import List, Sequence


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""

    return " ".join(text.lower().strip().split())


def _tokens(text: str) -> List[str]:
    return normalize_text(text).split()


class Metric:
    """Base class for metrics.

    A metric maps two equal length sequences of strings (predictions and
    references) to a single float score in [0, 1], where higher is better.
    """

    name: str = "metric"
    higher_is_better: bool = True

    def score(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> float:
        raise NotImplementedError

    def _check(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> None:
        if len(predictions) != len(references):
            raise ValueError(
                "predictions and references must have equal length, "
                f"got {len(predictions)} and {len(references)}"
            )
        if len(predictions) == 0:
            raise ValueError("cannot score an empty sequence")


class ExactMatchAccuracy(Metric):
    """Fraction of predictions that exactly match the reference after normalization."""

    name = "exact_match"
    higher_is_better = True

    def score(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> float:
        self._check(predictions, references)
        hits = sum(
            normalize_text(p) == normalize_text(r)
            for p, r in zip(predictions, references)
        )
        return hits / len(predictions)


class TokenF1(Metric):
    """Mean token level F1 between predictions and references.

    For each pair the score is the harmonic mean of token precision and recall,
    counting shared tokens with multiplicity. The metric returns the average
    over all pairs.
    """

    name = "token_f1"
    higher_is_better = True

    @staticmethod
    def _pair_f1(prediction: str, reference: str) -> float:
        pred_tokens = _tokens(prediction)
        ref_tokens = _tokens(reference)
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # Count shared tokens with multiplicity.
        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        shared = 0
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                shared += 1
                ref_counts[token] -= 1

        if shared == 0:
            return 0.0
        precision = shared / len(pred_tokens)
        recall = shared / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def score(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> float:
        self._check(predictions, references)
        total = sum(
            self._pair_f1(p, r) for p, r in zip(predictions, references)
        )
        return total / len(predictions)


_METRIC_REGISTRY = {
    ExactMatchAccuracy.name: ExactMatchAccuracy,
    TokenF1.name: TokenF1,
}


def metric_from_name(name: str) -> Metric:
    """Look up and instantiate a metric by its registered name."""

    if name not in _METRIC_REGISTRY:
        available = ", ".join(sorted(_METRIC_REGISTRY))
        raise KeyError(f"unknown metric '{name}', available: {available}")
    return _METRIC_REGISTRY[name]()
