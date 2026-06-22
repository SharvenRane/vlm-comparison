import torch

from vlm_comparison.adapters import (
    ConstantAdapter,
    KeywordMatchAdapter,
    NearestExampleAdapter,
    Prediction,
)
from vlm_comparison.benchmark import BenchmarkItem, build_default_benchmark


def _color_item(item_id, color, answer):
    image = torch.zeros(3, 4, 4)
    for c, v in enumerate(color):
        image[c].fill_(v)
    return BenchmarkItem(
        item_id=item_id,
        image=image,
        prompt="What is the dominant color in this image?",
        answer=answer,
        category="color",
    )


def test_constant_adapter_ignores_input():
    adapter = ConstantAdapter("blue", name="always-blue")
    item = _color_item("a", (1.0, 0.0, 0.0), "red")
    assert adapter.predict_one(item) == "blue"
    assert adapter.name == "always-blue"


def test_keyword_adapter_reads_color_from_image():
    adapter = KeywordMatchAdapter()
    assert adapter.predict_one(_color_item("r", (1.0, 0.0, 0.0), "red")) == "red"
    assert adapter.predict_one(_color_item("g", (0.0, 1.0, 0.0), "green")) == "green"
    assert adapter.predict_one(_color_item("b", (0.0, 0.0, 1.0), "blue")) == "blue"


def test_keyword_adapter_uses_prompt_for_text_questions():
    adapter = KeywordMatchAdapter()
    item = BenchmarkItem(
        item_id="cats",
        image=torch.full((3, 4, 4), 0.3),
        prompt="How many cats are in the picture?",
        answer="two cats",
    )
    assert adapter.predict_one(item) == "two cats"


def test_predict_returns_predictions_in_order():
    adapter = ConstantAdapter("x")
    items = [
        _color_item("a", (1.0, 0.0, 0.0), "red"),
        _color_item("b", (0.0, 1.0, 0.0), "green"),
    ]
    preds = adapter.predict(items)
    assert [p.item_id for p in preds] == ["a", "b"]
    assert all(isinstance(p, Prediction) for p in preds)


def test_nearest_example_copies_nearest_support_answer():
    support = [
        _color_item("s-red", (1.0, 0.0, 0.0), "red"),
        _color_item("s-green", (0.0, 1.0, 0.0), "green"),
    ]
    adapter = NearestExampleAdapter(support)
    query = _color_item("q", (0.9, 0.05, 0.05), "red")
    assert adapter.predict_one(query) == "red"


def test_keyword_adapter_strong_on_default_benchmark():
    bench = build_default_benchmark()
    adapter = KeywordMatchAdapter()
    preds = adapter.predict(list(bench))
    by_id = {p.item_id: p.text for p in preds}
    correct = sum(by_id[item.item_id] == item.answer for item in bench)
    # The keyword adapter is designed to solve the whole default benchmark.
    assert correct == len(bench)
