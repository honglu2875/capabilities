import math
from typing import Generic, Iterable, Literal, Optional, TypeVar, get_args
from .search_index import AbstractSearchIndex, SearchResult, get_chunks
from .types import EmbeddingModel, TextItem, get_text, Chunk
import numpy as np

try:
    from nomic import atlas
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use Nomic vector search, please install the python package using `pip install nomic`"
    )

T = TypeVar("T", bound=TextItem)


class NomicIndex(Generic[T], AbstractSearchIndex[T]):
    """Search index using Nomic Atlas."""

    project: atlas.AtlasProject
    modality: Literal["text", "embedding"]
    items: dict[str, T]
    chunks: dict[str, Chunk]

    def __init__(
        self,
        *,
        embedding_model: Optional[EmbeddingModel] = None,
        project_name: str,
        items: Optional[Iterable[T]] = None,
        reset_project_if_exists: bool = True,
        **kwargs,
    ):
        """
        Creates a new Nomic index.

        Note that this will create a new Nomic Atlas project if one does not already exist.
        In the case that the index already exists, the data will be wiped from it unless reset_project_if_exists is set to False.

        Before initialising `NomicIndex`, please make sure that you have logged in to Nomic, you can do this by typing `nomic login` into the terminal.


        Args:
          - project_name (str): name of the nomic project to use
          - embedding_model (EmbeddingModel, optional): the model to use for the index. Default is SentenceTransformer miniLM.
          - reset_project_if_exists (bool = False): whether to reset the project if it already exists. Defaults to True.
          - items (Iterable[T], optional): an iterable of items that will be used to update the index. If this is set, equivalent to calling `index.update(items)` after initialization.
        """
        if embedding_model is None:
            from .hf import STEmbeddingModel

            embedding_model = STEmbeddingModel()
        self.embedding_model = embedding_model
        self.modality = "text" if embedding_model is None else "embedding"
        self.project = atlas.AtlasProject(
            name=project_name,
            **kwargs,
            modality=self.modality,
            unique_id_field="id",
            reset_project_if_exists=reset_project_if_exists,
        )
        self.project_id = self.project.id
        self.items = {}
        self.chunks = {}
        if items is not None:
            self.update(items)

    _state_keys = ["items", "chunks", "project_id", "embedding_model", "modality"]

    def __getstate__(self):
        # [todo] might be able to get away with using the items and chunks stored on nomic?
        return {k: getattr(self, k) for k in self._state_keys}

    def __setstate__(self, state):
        for k in self._state_keys:
            setattr(self, k, state[k])
        self.project = atlas.AtlasProject(
            project_id=self.project_id,
            reset_project_if_exists=False,
        )

    def get_item(self, item_id: str) -> T:
        return self.items[item_id]

    def item_ids(self) -> Iterable[str]:
        return self.items.keys()

    @property
    def index(self):
        if len(self.chunks) < 20:
            raise RuntimeError(
                "Nomic does not yet have enough data to index. Please make sure there are at least 20 datapoints."
            )
        if len(self.project.indices) == 0:
            return self.project.create_index(
                name="main-index", colorable_fields=["item_id"]
            )
        else:
            return self.project.indices[0].projections[0]  # idk

    def update(self, items: Iterable[T]):
        items = list(items)
        for item in items:
            if item.id in self.items:
                raise RuntimeError(f"Item {item.id} already exists in the index.")
            self.items[item.id] = item
        if self.embedding_model is not None:
            chunks = list(get_chunks(items, self.embedding_model))
            for chunk in chunks:
                if chunk.unique_id in self.chunks:
                    raise RuntimeError(
                        f"Chunk {chunk.unique_id} already exists in the index."
                    )
                self.chunks[chunk.unique_id] = chunk
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            assert isinstance(embeddings, np.ndarray)
            assert len(embeddings.shape) == 2
            self.project.add_embeddings(
                data=[c.dict() for c in chunks],
                embeddings=embeddings,
            )
        else:
            self.project.add_text([Chunk.total(item).dict() for item in items])

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult[T]]:
        if self.embedding_model is not None:
            embedding = self.embedding_model.encode([query])
            idx = self.index
            with self.project.wait_for_project_lock():
                # note, Nomic scores using distance: lower is closer.
                ids, distances = idx.vector_search(queries=embedding, k=limit)
                assert len(ids) == embedding.shape[0] == len(distances)

                def mk(id, distance):
                    chunk = self.chunks[id]
                    item = self.items[chunk.item_id]
                    # invert score with atan to get a score clamped between 0 and 1 that increases with similarity
                    score = 1.0 - (math.atan(distance) * 2.0 / math.pi)
                    return SearchResult(
                        item=item,
                        score=score,
                        chunk_id=chunk.chunk_id,
                        substring_range=chunk.substring_range,
                    )

                items = [mk(id, score) for id, score in zip(ids[0], distances[0])]
                return items

        else:
            raise NotImplementedError(
                "search on a text-mode Nomic index is not yet implemented."
            )

    def __len__(self):
        raise NotImplementedError("todo")
