from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import itertools
import logging
import math
import re
from typing import Any, Dict, Generic, Iterable, Optional, Tuple, TypeVar, Union
from .simple_vector_index import SimpleVectorIndex
from .types import (
    Chunk,
    EmbeddingModel,
    TextItem,
    VectorIndex,
    get_text,
)
from .util import Bijection, argwindow, unzip
import numpy as np

T = TypeVar("T", bound=TextItem)

logger = logging.getLogger("capabilities.search")


def simple_chunker(
    item: TextItem, max_tokens_per_chunk: int, model: EmbeddingModel
) -> Iterable[Chunk]:
    """Takes a TextItem, and chunks it so that each chunk has at most max_tokens_per_chunk tokens.

    Args:
        item: TextItem to chunk.
        max_tokens_per_chunk: Maximum number of tokens per chunk.
        model: EmbeddingModel's tokenizer is used to figure out how many tokens are in a chunks.
    """
    N = max_tokens_per_chunk
    if N is None:
        yield Chunk.total(item)
        return
    text = get_text(item)
    tokens = model.tokenize(text)
    if len(tokens) < N:
        yield Chunk.total(item)
        return
    # [todo] maybe support a model.n_tokens_offset(n_tokens) -> offset method?
    ERROR_MARGIN = 10
    assert (
        ERROR_MARGIN < N
    ), f"max_tokens_per_chunk {N} should be larger than {ERROR_MARGIN}"
    windows = list(argwindow(len(tokens), N - ERROR_MARGIN, 0.5))

    for window_start, window_end in windows:
        chunk_text = model.detokenize(tokens[window_start:window_end])
        pre_chunk_text = model.detokenize(tokens[:window_start])
        offset_start = len(pre_chunk_text)
        offset_end = len(pre_chunk_text) + len(chunk_text)
        # assert offset_start < len(text)
        # assert offset_end <= len(text)
        assert model.count_tokens(chunk_text) <= N
        if chunk_text != text[offset_start:offset_end]:
            logger.debug("detokenization doesn't match.")
        yield Chunk(
            item_id=item.id,
            chunk_id=f"[{offset_start}:{offset_end}]",
            text=chunk_text,
            substring_range=range(offset_start, offset_end),
        )


class ChunkMap:
    """Keeps track of the mappings between chunk_ids, ids, indexes."""

    def __init__(self):
        self._idx_of_id: Dict[str, dict[Optional[str], int]] = defaultdict(dict)
        self._id_of_idx: Dict[int, str] = {}
        self._chunk_id_of_idx: Dict[int, Optional[str]] = {}
        self._chunk_range_of_chunk_id: dict[
            tuple[str, Optional[str]], Optional[range]
        ] = {}

    def __len__(self):
        return len(self._id_of_idx)

    def add(self, chunk: Chunk):
        index = len(self)
        idx_of_chunk_id = self._idx_of_id[chunk.item_id]
        assert chunk.chunk_id not in idx_of_chunk_id
        idx_of_chunk_id[chunk.chunk_id] = index
        assert index not in self._id_of_idx
        self._id_of_idx[index] = chunk.item_id
        assert index not in self._chunk_id_of_idx
        self._chunk_id_of_idx[index] = chunk.chunk_id
        k = (chunk.item_id, chunk.chunk_id)
        assert k not in self._chunk_range_of_chunk_id
        self._chunk_range_of_chunk_id[k] = chunk.substring_range
        return chunk.text

    def extend(self, chunks: Iterable[Chunk]) -> Iterable[str]:
        for chunk in chunks:
            yield self.add(chunk)

    def id_of_idx(self, idx: int):
        return self._id_of_idx[idx]

    def idxs_of_id(self, id: str):
        return list(self._idx_of_id[id].values())

    def chunk_id_of_idx(self, idx):
        return self._chunk_id_of_idx.get(idx, None)

    def idx_of_id(self, id: str, chunk_id: Optional[str] = None):
        return self._idx_of_id[id][chunk_id]

    def is_chunked(self, id: str):
        return None in self._idx_of_id[id]

    def chunk_range(self, id: str, chunk_id: str):
        return self._chunk_range_of_chunk_id[(id, chunk_id)]


T = TypeVar("T", bound=TextItem)


@dataclass
class SearchResult(Generic[T]):
    """Result of calling SearchIndex.search."""

    item: T
    """ The TextItem that was found.

    Note that the item may have been chunked, in which case the chunk_id will be non-None.
    You can use item.get_text() to get the chunk text."""
    chunk_id: Optional[str]
    """ If the original text-item was too long and got chunked, we also pass an identifier for the chunk. """
    score: float
    """ Value between 0.0 and 1.0 representing the score of the item, higher is closer to the query. """
    substring_range: Optional[range] = field(default=None)

    @property
    def id(self):
        """Gets the item_id of the found item."""
        return self.item.id

    def get_text(self) -> str:
        """Gets the chunk text of the found item."""
        fulltext = self.item.get_text()
        if self.substring_range is not None:
            r = self.substring_range
            return fulltext[r.start : r.stop]
        else:
            return fulltext


class AbstractSearchIndex(Generic[T], ABC):
    @abstractmethod
    def update(self, items: Iterable[T]):
        raise NotImplementedError()

    @abstractmethod
    def search(self, query: str, limit=5) -> list[SearchResult[T]]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def item_ids(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_item(self, item_id: str) -> T:
        """Get the item for the given id. Raises KeyError if not found."""
        raise NotImplementedError()


def chunk_item(
    item: TextItem, max_chunk_size: int, model: EmbeddingModel
) -> Iterable[Chunk]:
    """Run item.chunk or fallback to simple_chunker.

    Also performs some validation that the chunks are constructed correctly.
    """
    chunks = item.chunk(max_chunk_size, model)
    if chunks is NotImplemented:
        chunks = simple_chunker(item, max_chunk_size, model)
    chunks = list(chunks)
    assert len(set(c.chunk_id for c in chunks)) == len(chunks), "Duplicate chunk ids"
    for chunk in chunks:
        assert chunk.item_id == item.id, "invalid item_id on chunk"
    return chunks


def get_chunks(items: Iterable[TextItem], model: EmbeddingModel) -> Iterable[Chunk]:
    for item in items:
        text = get_text(item)
        total = model.count_tokens(text)
        N = model.max_tokens_per_item
        if N is not None and (total > N):
            yield from chunk_item(item, N, model)
        else:
            yield Chunk.total(item)


class SearchIndex(Generic[T], AbstractSearchIndex[T]):
    """A SearchIndex is a reference to a persistent vector database store.

    The store can be either local or on a cloud service.
    The params to __init__ should be enough to uniquely resolve the resource.

    Items are added to the index using the `update` method.
    In order to add an item to the search index, the item needs to have an `id` property
    and a 'get_text' method.
    You can also pass many text-like Python objects to `update` without needing to wrap in a TextItem.
    Eg strings will be added as-is, with the id being a blake2 hash of the string.
    A `Path` to a text file, and URLs to HTML pages are also supported out-of-the-box.
    Finally, you can also pass a `dict[str, str]` to `update`.

    If the text returned by `get_text` is too long for the given encoding token limit.
    The text will be chunked into overlapping chunks that fit the context length.
    You can customise the chunking algorithm by overriding the `.chunk(token_length : int, tokenizer)` method, which should return
    a sequence of `Chunk`s.
    Customising chunking is useful in cases where there is extra metadata (such as file name) that you want to include with all chunks.
    Or if the domain has clear points (eg paragraph boundaries) where chunking makes sense.

    Note that the index is just an index and does not store a copy of the text that creates the embeddings.
    You should have your own persistence layer for mapping Ids.

    """

    vector_index: VectorIndex
    embedding_model: EmbeddingModel
    _cmap: ChunkMap
    items: dict[str, T]

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_index: Optional[VectorIndex] = None,
        items: Optional[Iterable[T]] = None,
    ):
        if embedding_model is None:
            from capabilities.search.hf import STEmbeddingModel

            embedding_model = STEmbeddingModel()
        self.embedding_model = embedding_model
        self.vector_index = vector_index or SimpleVectorIndex()
        self._cmap = ChunkMap()
        self.items = {}
        if items is not None:
            self.update(items)

    def update(self, items: Iterable[T]) -> None:
        """Adds the given items to the index or replaces them if they already exist.

        If the text returned by `get_text` is too long for the given encoding token limit
        (= self.embedding_model.max_tokens_per_item) then this function will automatically chunk the text into overlapping chunks that fit the context length.

        You can customise the chunking algorithm by overriding the `.chunk(token_length : int, model)` method.
        The default algorithm is `simple_chunker`.

        Args:
            items (Iterable[TextItem]): The items to add to the index.
        Raises:
            NotImplementedError: If updating items that already exist in the index. (we are working on it)
        """
        items = list(items)
        for item in items:
            if item.id in self.items:
                # [todo]
                raise NotImplementedError(
                    "Updating items that already exist in the index is not currently supported."
                )
            self.items[item.id] = item
        texts = list(self._cmap.extend(get_chunks(items, self.embedding_model)))
        embeddings: np.ndarray = self.embedding_model.encode(texts)
        assert len(texts) == len(embeddings)
        self.vector_index.add(embeddings)
        assert len(self.vector_index) == len(self._cmap)
        logger.debug(
            f"Added {len(items)} items ({len(texts)} chunks) to the search index."
        )

    def search(self, query: str, limit=5) -> Iterable[SearchResult]:
        """Finds a set of nearest results for the given query.

        The exact algorithm used is determined by the choice of self.vector_index.
        The resulting SearchResult objects can be used to identify the documents and chunks of documents that match the query.

        SearchResult.id gives the id of the TextItem that matched the query.
        SearchResult.score gives the score of the match (eg cosine distance).

        If the TextItem was chunked, then two additional fields are available:
        - SearchResult.chunk_id is an id that uniquely identifies the chunk for a given TextItem.
        - SearchResult.substring_range is a substring range from the original TextItem's text that can be used to recover the original text that was matched.

        """
        encoding = self.embedding_model.encode_one(query)
        d, i = self.vector_index.search_one(encoding, limit=limit)

        def mk(score, index):
            id = self._cmap.id_of_idx(index)
            chunk_id = self._cmap.chunk_id_of_idx(index)
            if chunk_id is not None:
                chunk_range = self._cmap.chunk_range(id, chunk_id)
            else:
                chunk_range = None
            item = self.items[id]

            return SearchResult(
                item=item,
                score=float(score),
                chunk_id=chunk_id,
                substring_range=chunk_range,
            )

        results = [mk(score, index) for score, index in zip(d, i)]
        return results

    def __len__(self):
        return len(self.vector_index)

    def get_item(self, item_id: str) -> T:
        return self.items[item_id]

    def item_ids(self) -> Iterable[str]:
        return self.items.keys()
