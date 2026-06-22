"""Model adapters for the comparison harness.

An adapter wraps a model behind a uniform interface so the harness can treat
every model the same way. A real adapter would call into LLaVA, PaliGemma, or a
hosted endpoint. The adapters here are deterministic stand ins that run offline,
but they exercise the same interface a real adapter would: they receive the
image tensor and the prompt and return a text prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .benchmark import BenchmarkItem


@dataclass(frozen=True)
class Prediction:
    """A single model output.

    Attributes:
        item_id: identifier of the benchmark item this answers.
        text: the predicted answer string.
    """

    item_id: str
    text: str


class ModelAdapter:
    """Base class for all adapters.

    Subclasses implement :meth:`predict_one`. The harness only relies on
    :meth:`name` and :meth:`predict`, so a real adapter that batches calls to a
    server can override :meth:`predict` directly.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("adapter name must be non empty")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def predict_one(self, item: BenchmarkItem) -> str:
        raise NotImplementedError

    def predict(self, items: Sequence[BenchmarkItem]) -> List[Prediction]:
        """Run the model over a sequence of items in order."""

        return [
            Prediction(item_id=item.item_id, text=self.predict_one(item))
            for item in items
        ]


class ConstantAdapter(ModelAdapter):
    """An adapter that always returns the same string.

    This is a useful weak baseline. It ignores both the image and the prompt.
    """

    def __init__(self, constant: str, name: str = "constant") -> None:
        super().__init__(name)
        self._constant = constant

    def predict_one(self, item: BenchmarkItem) -> str:
        return self._constant


# A small lexicon mapping channel dominance to a color word. This lets a vision
# aware adapter answer color questions by reading the image tensor.
_CHANNEL_COLORS = ("red", "green", "blue")


class KeywordMatchAdapter(ModelAdapter):
    """A hybrid adapter that uses the image for color and the prompt otherwise.

    For color questions it inspects channel means and reports the dominant
    channel. For everything else it scans the prompt for known keywords and
    emits a canned answer. This stands in for a capable vision language model
    that can both see and read, without needing real weights.
    """

    def __init__(
        self,
        keyword_answers: Optional[Dict[str, str]] = None,
        name: str = "keyword-match",
    ) -> None:
        super().__init__(name)
        self._keyword_answers: Dict[str, str] = {
            "cat": "two cats",
            "dog": "three dogs",
            "beach": "at the beach",
            "forest": "in the forest",
        }
        if keyword_answers:
            self._keyword_answers.update(keyword_answers)

    def predict_one(self, item: BenchmarkItem) -> str:
        prompt = item.prompt.lower()
        if "color" in prompt:
            channel_means = item.image.reshape(item.image.shape[0], -1).mean(dim=1)
            dominant = int(torch.argmax(channel_means).item())
            return _CHANNEL_COLORS[dominant]
        for keyword, answer in self._keyword_answers.items():
            if keyword in prompt:
                return answer
        return "unknown"


class NearestExampleAdapter(ModelAdapter):
    """A retrieval style adapter backed by labeled support examples.

    It stores a few example items with known answers, embeds every image as its
    channel mean vector, and at inference time copies the answer of the nearest
    support image by Euclidean distance. This mirrors how a tiny nearest
    neighbor probe over frozen visual features would behave. It is deliberately
    weaker on text only questions because it relies on visual similarity.
    """

    def __init__(
        self,
        support_items: Sequence[BenchmarkItem],
        name: str = "nearest-example",
    ) -> None:
        super().__init__(name)
        if len(support_items) == 0:
            raise ValueError("support_items must be non empty")
        self._support_answers: List[str] = []
        embeddings = []
        for item in support_items:
            embeddings.append(self._embed(item.image))
            self._support_answers.append(item.answer)
        self._support_embeddings = torch.stack(embeddings, dim=0)

    @staticmethod
    def _embed(image: torch.Tensor) -> torch.Tensor:
        return image.reshape(image.shape[0], -1).mean(dim=1)

    def predict_one(self, item: BenchmarkItem) -> str:
        query = self._embed(item.image)
        distances = torch.linalg.norm(self._support_embeddings - query, dim=1)
        nearest = int(torch.argmin(distances).item())
        return self._support_answers[nearest]
