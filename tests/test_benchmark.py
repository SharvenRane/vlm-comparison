import pytest
import torch

from vlm_comparison.benchmark import (
    BenchmarkItem,
    Benchmark,
    build_default_benchmark,
)


def test_default_benchmark_is_non_empty():
    bench = build_default_benchmark()
    assert len(bench) >= 8


def test_default_benchmark_items_are_well_formed():
    bench = build_default_benchmark()
    for item in bench:
        assert isinstance(item.image, torch.Tensor)
        assert item.image.dim() == 3
        assert item.image.shape[0] == 3
        assert item.prompt
        assert item.answer
        assert float(item.image.min()) >= 0.0
        assert float(item.image.max()) <= 1.0


def test_color_items_encode_dominant_channel():
    bench = build_default_benchmark()
    channel_to_color = ("red", "green", "blue")
    for item in bench:
        if item.category != "color":
            continue
        means = item.image.reshape(3, -1).mean(dim=1)
        dominant = int(torch.argmax(means).item())
        assert channel_to_color[dominant] == item.answer


def test_item_ids_are_unique():
    bench = build_default_benchmark()
    ids = [item.item_id for item in bench]
    assert len(set(ids)) == len(ids)


def test_benchmark_answers_property_matches_items():
    bench = build_default_benchmark()
    assert bench.answers == [item.answer for item in bench]


def test_bad_image_dim_raises():
    with pytest.raises(ValueError):
        BenchmarkItem(
            item_id="x",
            image=torch.zeros(4, 4),
            prompt="q",
            answer="a",
        )


def test_empty_prompt_raises():
    with pytest.raises(ValueError):
        BenchmarkItem(
            item_id="x",
            image=torch.zeros(3, 4, 4),
            prompt="",
            answer="a",
        )


def test_non_tensor_image_raises():
    with pytest.raises(TypeError):
        BenchmarkItem(
            item_id="x",
            image=[[0.0]],
            prompt="q",
            answer="a",
        )


def test_benchmark_add_grows():
    bench = Benchmark()
    assert len(bench) == 0
    bench.add(
        BenchmarkItem(
            item_id="x",
            image=torch.zeros(3, 2, 2),
            prompt="q",
            answer="a",
        )
    )
    assert len(bench) == 1
