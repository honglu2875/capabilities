from capabilities.search import SearchIndex, create_document
from rich import print
import re
from rich.panel import Panel

links = ["https://en.wikipedia.org/wiki/Attitude_(heraldry)", "https://cheese.com"]

index = SearchIndex()
index.update(map(create_document, links))

results = list(index.search("gruyere"))

for result in results:
    doc = result.item
    # ↓ this will automatically extract the correct chunk of text.
    text = result.get_text()
    text = re.sub(r"\n", " ", text)  # tidy up excess newlines for printing
    print(result.id, result.score, Panel(text))
