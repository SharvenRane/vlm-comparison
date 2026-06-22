# vlm-comparison

A small harness for comparing vision language models on a benchmark. You plug in
model adapters and a metric, the harness runs every adapter over the benchmark,
scores the predictions, and hands you a ranking from best to worst.

The point of the project is the plumbing around the models, not the models
themselves. The adapters that ship here are deterministic stand ins so the whole
pipeline runs offline on CPU with no downloads and no API keys. Swap in a real
adapter for LLaVA or PaliGemma and the rest of the harness stays exactly the
same.

## What is inside

The package has four pieces that compose cleanly:

- **Benchmark.** A list of items. Each item carries a small synthetic image
  tensor, a text prompt, and the reference answer. The default benchmark mixes
  color questions, where the answer lives in the image, with caption questions,
  where the answer lives in the prompt.
- **Adapters.** Each adapter wraps a model behind one interface: take an item,
  return a predicted string. A real adapter would call a model or an endpoint.
  The bundled adapters read the image tensor or scan the prompt so the harness
  has something honest to rank.
- **Metrics.** Exact match accuracy and token level F1. Both take predictions
  and references as plain strings and return a score in zero to one where higher
  is better.
- **Harness.** The runner. It calls every adapter over the benchmark, aligns
  predictions to items by id, scores them, and sorts the adapters. Ties keep the
  input order so the ranking is deterministic.

## The bundled adapters

None of these need real weights, but each one exercises the interface a real
model would use.

- **KeywordMatchAdapter** behaves like a capable model that can both see and
  read. For color questions it inspects channel means and names the dominant
  channel. For other questions it scans the prompt for known keywords and
  returns a canned answer. It is built to solve the default benchmark.
- **NearestExampleAdapter** is a retrieval style baseline. You give it labeled
  support items, it embeds every image as its channel mean vector, and at
  inference it copies the answer of the nearest support image by Euclidean
  distance. This is how a tiny nearest neighbor probe over frozen features would
  behave, and it is weaker on text only questions because it leans on visual
  similarity.
- **ConstantAdapter** always returns the same string. It ignores the image and
  the prompt, which makes it a clean weak baseline at the bottom of the ranking.

## Quick start

```python
from vlm_comparison import (
    build_default_benchmark,
    KeywordMatchAdapter,
    ConstantAdapter,
    ExactMatchAccuracy,
    run_comparison,
)

bench = build_default_benchmark()

adapters = [
    KeywordMatchAdapter(),
    ConstantAdapter("red", name="always-red"),
]

report = run_comparison(adapters, bench, ExactMatchAccuracy())

for row in report.as_table():
    print(row["rank"], row["adapter"], round(row["score"], 3))

print("winner:", report.best.adapter_name)
```

## Plugging in a real model

Subclass `ModelAdapter`, give it a name, and implement `predict_one`. If your
model batches well, override `predict` instead and return a list of `Prediction`
objects. Here is the shape of a real adapter that talks to a server:

```python
from vlm_comparison import ModelAdapter

class MyServerAdapter(ModelAdapter):
    def __init__(self, client):
        super().__init__(name="my-server")
        self._client = client

    def predict_one(self, item):
        return self._client.caption(item.image, item.prompt)
```

The harness treats it the same as any bundled adapter, so you can rank a real
model against the offline baselines without touching the runner.

## Running the tests

The tests are property and behavior checks. They build tiny synthetic tensors,
confirm the adapters read the image and the prompt as intended, and verify that
the harness score equals a metric computed by hand on the same predictions.

```
python -m pytest tests/ -q
```

On this machine the suite reports 37 passed.

## Layout

```
src/vlm_comparison/   benchmark, adapters, metrics, harness
tests/                pytest suite
requirements.txt      torch and pytest
```
