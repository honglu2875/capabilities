import logging
from typing import Sequence
from .types import EmbeddingModel
from hashlib import blake2b
from .util import batched, cache

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "In order to use sentence-transformer modules, please run `pip install sentence-transformers`"
    )

"""
This file contains SentenceTransformer models.
"""

logger = logging.getLogger(__name__)


class STEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def __getstate__(self):
        return {"name": self.name}

    def __setstate__(self, state):
        self.name = state["name"]
        self.model = SentenceTransformer(self.name)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text))  # type: ignore

    def tokenize(self, text: str) -> list[int]:
        return self.model.tokenizer.encode(text)  # type: ignore

    def detokenize(self, tokens: list[int]) -> str:
        # first and last tokens are special tokens
        # [todo] filter special tokens instead
        tokens = tokens[1:-1]
        return self.model.tokenizer.decode(tokens)  # type: ignore

    @property
    def max_tokens(self):
        return None

    @property
    def max_tokens_per_item(self) -> int | None:
        return 256

    def encode_no_cache(self, texts: list[str], show_progress_bar=True):
        return self.model.encode(list(texts), show_progress_bar=show_progress_bar)

    def encode(self, sentences: list[str], show_progress_bar=True):
        h = blake2b()
        h.update(self.name.encode())
        for sentence in sentences:
            h.update(sentence.encode())
        digest = h.hexdigest()
        result = cache.get(digest)
        if result is not None:
            logger.debug(f"Using cached encoding")
            return result
        result = self.encode_no_cache(sentences, show_progress_bar=show_progress_bar)
        cache.set(digest, result)
        return result

    def get_token_offsets(self, text: str):
        tokens = self.model.tokenizer(
            text, return_offsets_mapping=True, return_special_tokens_mask=True
        )  # type: ignore
        # xs and ys are monotone offsets
        xs, ys = [], []
        acc = 0
        for i, (x, y) in enumerate(tokens.offset_mapping):
            acc = max(acc, x)
            xs.append(acc)
            acc = max(acc, y)
            ys.append(acc)
        assert list(sorted(xs)) == xs
        assert list(sorted(ys)) == ys
        return xs, ys
