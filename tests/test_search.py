from pathlib import Path
import pprint
from dataclasses import asdict
import pickle

import pytest
from capabilities.search import AbstractSearchIndex, create_document, EmbeddingModel
from capabilities.search.models.hf import STEmbeddingModel
from capabilities.search.models.oai import OpenAIEmbeddingModel
from capabilities.search import SearchIndex
from capabilities.search.nomic_index import NomicIndex


@pytest.fixture(
    scope="module", params=["sentence-transformers/all-MiniLM-L6-v2", "openai"]
)
def embedding_model(request):
    param = request.param
    if param.startswith("sentence-transformers/"):
        yield STEmbeddingModel(param)
    elif param == "openai":
        # [todo] if no api key skip this one.
        yield OpenAIEmbeddingModel()
    else:
        raise ValueError(f"Unknown param {param}")


@pytest.fixture(scope="module", params=["SearchIndex", "NomicIndex"])
def search_index(request, embedding_model):
    param = request.param
    if param == "SearchIndex":
        yield SearchIndex(
            embedding_model=embedding_model,
        )
    elif param == "NomicIndex":
        idx = NomicIndex(embedding_model=embedding_model, project_name="test_project")
        yield idx
        idx.project.delete()
    else:
        raise ValueError(f"Unknown param {param}")


def test_search_e2e(search_index: AbstractSearchIndex, snapshot, tmp_path):
    data_dir = Path("examples/data")
    search_index.update([create_document(data_dir / "tesla10k.txt")])
    search_index.update([create_document(data_dir / "apple10k.pdf")])
    results = list(search_index.search("Kimbal", limit=5))
    assert len(results) == 5
    r0 = results[0]
    r = pprint.pformat(asdict(r0))
    # snapshot.assert_match(r, "r0.txt")

    # now save to disk
    snap_path = tmp_path / "snapshot.pkl"
    with open(snap_path, "wb") as f:
        pickle.dump(search_index, f)

    with open(snap_path, "rb") as f:
        index2 = pickle.load(f)

    assert isinstance(index2, type(search_index))

    r0_2 = list(index2.search("Kimbal", limit=5))[0]
    assert r0.get_text() == r0_2.get_text()


# [todo] test saving and restoring to disk
# [todo] if it's a cloud index, make sure that it doesn't get wiped.
