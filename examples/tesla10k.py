"""

In this example, we are going to make a QA bot for big text documents.


"""
import asyncio
import gzip
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel
from capabilities.search import *
import ndjson
from rich import print
from rich.prompt import Prompt
from rich.panel import Panel
from capabilities.search.models.hf import STEmbeddingModel
from capabilities.search.models.oai import OpenAIEmbeddingModel
from pathlib import Path
from capabilities.search.loader import create_document, Document
from capabilities.search.nomic_index import NomicIndex

# %%
# First lets create an index.


data_dir = Path("examples/data")
index = NomicIndex[Document](
    embedding_model=STEmbeddingModel(), project_name="tesla10k"
)

index.update([create_document(data_dir / "tesla10k.txt")])
index.update([create_document(data_dir / "apple10k.pdf")])

while True:
    try:
        query = Prompt.ask(
            "\n\n search the doc",
            default="What is Kimbal doing?",
            show_default=True,
        )
    except (KeyboardInterrupt, EOFError):
        break
    results = index.search(query)
    for result in results:
        doc = result.item
        chunk_id = result.chunk_id
        r = result.substring_range
        assert r is not None
        assert chunk_id is not None
        text = result.get_text()
        text = re.sub(r"\n", " ", text)  # tidy up
        print(f"[bold blue]{result.id}/{chunk_id}[/], ", "score =", result.score)
        print()
        print(Panel(text))
        print()
# %%
