"""Benchmark definition for the comparison harness.

A benchmark is a list of items. Each item carries a small synthetic image
(a torch tensor shaped channels by height by width), a text prompt, and the
reference answer the model is expected to produce.

The images here are deterministic synthetic tensors rather than real photos so
that everything stays offline and reproducible. An adapter is free to look at
the image, the prompt, or both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch


@dataclass(frozen=True)
class BenchmarkItem:
    """A single evaluation example.

    Attributes:
        item_id: stable identifier for the example.
        image: float tensor shaped (channels, height, width) in the range [0, 1].
        prompt: the question or instruction text shown to the model.
        answer: the reference answer string.
        category: a coarse label used for grouping (for example "color").
    """

    item_id: str
    image: torch.Tensor
    prompt: str
    answer: str
    category: str = "general"

    def __post_init__(self) -> None:
        if not isinstance(self.image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if self.image.dim() != 3:
            raise ValueError(
                f"image must have 3 dims (C, H, W), got shape {tuple(self.image.shape)}"
            )
        if not self.prompt:
            raise ValueError("prompt must be a non empty string")
        if not self.answer:
            raise ValueError("answer must be a non empty string")


@dataclass
class Benchmark:
    """An ordered collection of benchmark items."""

    items: List[BenchmarkItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> BenchmarkItem:
        return self.items[index]

    @property
    def answers(self) -> List[str]:
        return [item.answer for item in self.items]

    @property
    def categories(self) -> List[str]:
        return [item.category for item in self.items]

    def add(self, item: BenchmarkItem) -> None:
        self.items.append(item)


def _solid_image(color: Sequence[float], size: int = 8) -> torch.Tensor:
    """Build a solid color image tensor shaped (3, size, size).

    The color is a triple of red, green, blue intensities in [0, 1]. The mean
    over each channel equals the requested intensity, which lets an adapter
    recover the dominant color by inspecting channel means.
    """

    if len(color) != 3:
        raise ValueError("color must have three components")
    image = torch.zeros(3, size, size, dtype=torch.float32)
    for channel, value in enumerate(color):
        image[channel].fill_(float(value))
    return image


def build_default_benchmark() -> Benchmark:
    """Construct the default offline benchmark.

    The benchmark mixes two task families. Color items ask for the dominant
    color of a solid swatch, which a vision aware adapter can read from the
    image. Caption items ask for a short answer that depends only on the prompt,
    which a text aware adapter can answer by matching against known examples.

    The reference answers are intentionally simple lowercase strings so that the
    metrics stay easy to verify by hand.
    """

    bench = Benchmark()

    color_specs = [
        ("color-red", (1.0, 0.0, 0.0), "red"),
        ("color-green", (0.0, 1.0, 0.0), "green"),
        ("color-blue", (0.0, 0.0, 1.0), "blue"),
        ("color-dark-red", (0.7, 0.1, 0.1), "red"),
        ("color-dark-green", (0.1, 0.7, 0.1), "green"),
        ("color-dark-blue", (0.1, 0.1, 0.7), "blue"),
    ]
    for item_id, color, answer in color_specs:
        bench.add(
            BenchmarkItem(
                item_id=item_id,
                image=_solid_image(color),
                prompt="What is the dominant color in this image?",
                answer=answer,
                category="color",
            )
        )

    caption_specs = [
        ("count-two", "two", "How many cats are in the picture?", "two cats"),
        ("count-three", "three", "How many dogs are in the picture?", "three dogs"),
        (
            "scene-beach",
            "beach",
            "Was this beach photo taken by the ocean?",
            "at the beach",
        ),
        (
            "scene-forest",
            "forest",
            "Was this forest photo taken among the trees?",
            "in the forest",
        ),
    ]
    for item_id, tint_key, prompt, answer in caption_specs:
        # The caption images are mild gray tints so they carry no color signal.
        tint = 0.3 + 0.05 * (hash(tint_key) % 5)
        bench.add(
            BenchmarkItem(
                item_id=item_id,
                image=_solid_image((tint, tint, tint)),
                prompt=prompt,
                answer=answer,
                category="caption",
            )
        )

    return bench
