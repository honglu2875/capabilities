from typing import Generic, Iterable, Literal, Optional, TypeVar, get_args
from .search_index import AbstractSearchIndex, SearchResult, get_chunks
from .types import TextItem, get_text, Chunk
import numpy as np
from nomic import atlas

T = TypeVar("T", bound=TextItem)


class NomicIndex(Generic[T], AbstractSearchIndex[T]):
    project: atlas.AtlasProject
    modality: Literal["text", "embedding"]
    items: dict[str, T]
    chunks: dict[str, Chunk]

    def __init__(
        self,
        *,
        embedding_model=None,
        project_name: str,
        map_name="index",
        items: Optional[Iterable[T]] = None,
        **kwargs,
    ):
        self.embedding_model = embedding_model
        self.modality = "text" if embedding_model is None else "embedding"
        self.project = atlas.AtlasProject(
            name=project_name,
            **kwargs,
            modality=self.modality,
            unique_id_field="id",
            # [todo] should it reset?
            reset_project_if_exists=True,
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
        # if self.project.total_datums < 20:
        #     raise RuntimeError(
        #         "Nomic does not yet have enough data to index. Please make sure there are at least 20 datapoints."
        #     )
        if len(self.project.indices) == 0:
            return self.project.create_index(
                name="an-index", colorable_fields=["item_id"]
            )
        else:
            return self.project.indices[0].projections[0]  # idk

    def update(self, items: Iterable[T]):
        items = list(items)
        self.items.update({item.id: item for item in items})
        if self.embedding_model is not None:
            chunks = list(get_chunks(items, self.embedding_model))
            self.chunks.update({c.unique_id: c for c in chunks})
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
    ):
        if self.embedding_model is not None:
            embedding = self.embedding_model.encode([query])
            idx = self.index
            with self.project.wait_for_project_lock():
                ids, scores = idx.vector_search(queries=embedding, k=limit)
                for id, score in zip(ids[0], scores[0]):
                    chunk = self.chunks[id]
                    item = self.items[chunk.item_id]
                    yield SearchResult(
                        item=item,
                        score=score,  # type: ignore
                        chunk_id=chunk.chunk_id,
                        substring_range=chunk.substring_range,
                    )
        else:
            raise NotImplementedError("doesn't seem to be a direct text search api")

    def __len__(self):
        raise NotImplementedError("todo")
