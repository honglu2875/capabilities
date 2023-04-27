"""
Theorem Semantic Search
=======================

This example takes mathlib and produces a semantic search vector database. Nice.

"""

import asyncio
import gzip
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
import ndjson
from rich import print
from rich.prompt import Prompt
from rich.panel import Panel
from capabilities.search import SearchIndex, TextItem
from capabilities.search.hf import STEmbeddingModel
from capabilities.search.nomic_index import NomicIndex

# %%
# Preliminaries: download mathlib.jsonl.gz

mathlib_decls_url = "https://drive.google.com/file/d/1N3qrxx0vHRDUeTWESuXFgabqCqF3I5tE/view?usp=share_link"
mathlib_decls_path = Path("examples/downloads/mathlib.jsonl.gz")
if not mathlib_decls_path.exists():
    raise FileNotFoundError(
        f"Please visit {mathlib_decls_url} and store the file at {mathlib_decls_path}"
    )

# %%
# Define a new TextItem class for mathlib declarations.

class MathlibDecl(BaseModel, TextItem):
    formal_statement: str
    name: str
    doc_string: str

    def get_text(self):
        """Used as the text to get embedding from."""
        if self.doc_string:
            return f"/-- {self.doc_string} -/\n{self.formal_statement}"
        else:
            return self.formal_statement

    @property
    def id(self):
        return self.name

# %%
# Loading the data
# ----------------
#
# Let's grab the data from a file on gdrive.

with gzip.open(mathlib_decls_path, "rt") as f:
    mathlib_decls = list(map(MathlibDecl.parse_obj, ndjson.reader(f)))

# mathlib_decls = mathlib_decls[:1000]

print(f"Found {len(mathlib_decls)} mathlib decls")

# %%
# Now let's create a semantic search vector database.

vdb = NomicIndex[MathlibDecl](
    embedding_model=STEmbeddingModel(),
    items=mathlib_decls,
    project_name="leansearch",
)


# %%
# Now we can run a little event loop thing.


while True:
    try:
        query = Prompt.ask(
            "\n\n Name some mathematics: ",
            default="Second isomorphism theorem",
            show_default=True,
        )
    except (KeyboardInterrupt, EOFError):
        break
    results = vdb.search(query)
    for result in results:
        decl = result.item
        print(
            f"[bold blue]{decl.name}",
        )
        print("score =", result.score)
        print(Panel(decl.get_text()))
        print()
