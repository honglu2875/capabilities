from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from functools import cached_property
import functools
from typing import (
    Dict,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from capabilities.search.util import digest
import numpy as np


@dataclass
class Chunk:
    """A Chunk is a subset of a TextItem.

    Chunking is required when a piece of text is too long for the embedding model.
    """

    item_id: str
    chunk_id: Optional[str]
    text: str
    substring_range: Optional[range] = field(default=None)
    """ Subrange of the original TextItem text that this chunk covers.
    Note that chunk.text does not necessarily equal to the substring:
    chunks can include additional context in their text like titles and filenames.
    """

    @classmethod
    def total(cls, item: "TextItem"):
        """Return the 'everything' chunk for the given TextItem."""
        text = get_text(item)
        return cls(
            item_id=item.id,
            chunk_id=None,
            text=text,
            substring_range=range(len(text)),
        )

    @property
    def unique_id(self):
        # note: nomic does not like having long ids.
        return digest(f"Chunk({self.item_id}, {self.chunk_id})")[:36]

    def dict(self):
        """Dict for feeding into databases like nomic."""
        return {
            "id": self.unique_id,
            "item_id": self.item_id,
            "chunk_id": self.chunk_id if self.chunk_id is not None else "none",
            "text": self.text,
        }


class TextItem(ABC):
    """A TextItem is anything that can be converted to text and embedded.

    In order to be able to embed, you need to provide an id property and a get_text() method.

    If the string given in `get_text()` is too long for the embedding model, it will be chunked automatically.
    You can control how text should be chunked by providing a `chunk()` method.
    """

    @cached_property
    def digest(self) -> str:
        """Returns the blake2b digest of the item's text as a hex string."""
        return digest(self.get_text())

    @property
    def id(self) -> str:
        """Returns a unique identifier for the item.

        By default, this is the blake2b digest of the item's text.
        We recommend overriding this with an id that better reflects the semantics of the document.
        For example, if the TextItem represents a webpage, choose the id to be the URL of the page.
        """
        # note that 36 is chosen because nomic does not like having long ids.
        return self.digest[:36]

    @abstractmethod
    def get_text(self) -> str:
        """Returns the embeddable text."""
        raise NotImplementedError()

    def chunk(self, token_length: int, model: "EmbeddingModel") -> Iterable[Chunk]:
        return NotImplemented


@functools.singledispatch
def get_text(item) -> str:
    """Returns the embeddable text for the given item."""
    raise TypeError(f"unexpected {type(item)}")


@get_text.register(TextItem)
def _text_item_get_text(item: TextItem) -> str:
    return item.get_text()


@get_text.register(str)
def _str_get_text(text: str) -> str:
    return text


class EmbeddingModel(ABC):
    """Embedding inference model."""

    @cached_property
    def dim(self):
        """Get the encoding dimension."""
        return len(self.encode_one("hello world"))

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Run tokenizer on given text.

        This is used by the chunking algorithm to produce sufficiently small chunks.

        Returns:
          tokens: list[int] a list of token input ids.
        """
        raise NotImplementedError()

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Inverse of tokenizer on given list of tokens."""
        raise NotImplementedError()

    def count_tokens(self, text: str) -> int:
        """Gets the number of tokens that the given text will encode to.

        This is used to determine whether the text is too long and needs to be chunked.
        """
        return len(self.tokenize(text))

    def get_token_offsets(self, text: str) -> tuple[list[int], list[int]]:
        """Returns a pair of sequences giving the start and end string offset of each token.

        This is used by various chunking algorithms to ensure that
        the resulting chunks are within the embedding model's
        maximum token length per item limit.

        Args:
            text: input string to tokenize

        Returns:
           start: list[int]
           end: list[int]

        The following properties must be true:
        1. The length of start and end must be equal to `self.count_tokens(text)`
        2. start and end must both be monotonic sequences, that is `x[i] ≤ x[j]` for all `i ≤ j`.
        3. `start[i] ≤ end[i]` for all `i`.
        4. `text[start[i]:end[i]]` is a valid slicing (can be empty).

        """
        raise NotImplementedError()

    @abstractproperty
    def max_tokens_per_item(self) -> Optional[int]:
        """Maximum number of tokens allowed per text row passed to 'encode'.
        If None, then there is no limit.
        """
        return None

    def encode_one(self, text: str) -> np.ndarray:
        """
        Returns:
          np.ndarray with shape `(self.dim, )`, the encoding of the text.
        """
        x = self.encode([text])
        # assert x.shape == (1, self.dim)
        return x[0]

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encodes a sequence of texts.

        The encoding method will take care of all matters to do with
        batching, padding, rate limiting etc.
        But not making sure each text is less than the number of allowed tokens.

        Returns:
            an np.ndarray with shape `(len(texts), self.dim)`.
        """
        raise NotImplementedError()

    # [todo] async / streaming version


T = TypeVar("T", bound=TextItem)


class VectorIndex(ABC):
    @abstractmethod
    def add(self, arr: np.ndarray):
        raise NotImplementedError()

    @abstractproperty
    def dim(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def search(self, queries: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search the vector index.

        Args:
            - queries (np.ndarray): A batch of queries with shape (N, self.dim).
            - limit(int): the number of results to return.
        Returns:
            - distances: float np.ndarray with shape (N, limit)
            - indices: integer np.ndarray with shape (N, limit) giving the indices of the matching results.
        """
        raise NotImplementedError()

    def search_one(self, query: np.ndarray, limit=10) -> Tuple[np.ndarray, np.ndarray]:
        d, i = self.search(query[np.newaxis, :], limit)
        return d[0], i[0]

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
