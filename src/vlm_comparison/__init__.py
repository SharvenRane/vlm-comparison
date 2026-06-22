"""Vision language model comparison harness.

This package provides a small benchmark of vision language tasks, a set of
pluggable model adapters, pluggable metrics, and a runner that scores every
adapter over the benchmark and ranks them.

The adapters shipped here include deterministic stub models so that the whole
pipeline runs offline on CPU with no downloads and no API keys. The benchmark,
metric, and ranking logic around those stubs is real.
"""

from .benchmark import BenchmarkItem, Benchmark, build_default_benchmark
from .adapters import (
    ModelAdapter,
    Prediction,
    KeywordMatchAdapter,
    NearestExampleAdapter,
    ConstantAdapter,
)
from .metrics import (
    Metric,
    ExactMatchAccuracy,
    TokenF1,
    metric_from_name,
)
from .harness import (
    AdapterResult,
    ComparisonReport,
    run_comparison,
)

__all__ = [
    "BenchmarkItem",
    "Benchmark",
    "build_default_benchmark",
    "ModelAdapter",
    "Prediction",
    "KeywordMatchAdapter",
    "NearestExampleAdapter",
    "ConstantAdapter",
    "Metric",
    "ExactMatchAccuracy",
    "TokenF1",
    "metric_from_name",
    "AdapterResult",
    "ComparisonReport",
    "run_comparison",
]
