import logging
import os
import io
from pathlib import Path

from urllib.parse import urlparse
from urllib import request

from dataclasses import dataclass
from pdfminer.high_level import extract_text
from capabilities.search.util import digest
import magic
from bs4 import BeautifulSoup

from typing import Hashable, Optional, Union

from .types import TextItem

logger = logging.getLogger(__name__)


@dataclass
class Document(TextItem):
    text: str
    doc_id: str
    location: str
    digest: str

    @property
    def id(self):
        return str(self.doc_id)

    def get_text(self) -> str:
        return self.text


def create_document(location: Union[str, Path], doc_id: Optional[str] = None) -> Document:
    """Create a document from a web url or file path.

    Args:
      - location: is a web url, or a path to a pdf, md, or txt file.
      - doc_id: is a unique identifier for the document, if not given then the location string will be used.
    """
    location = str(location)
    if doc_id is None:
        doc_id = location
    parsed_url = urlparse(location)

    if parsed_url.scheme and parsed_url.netloc:
        response = request.urlopen(location)
        byte_content = response.read()
    elif os.path.isfile(location):
        with open(location, "rb") as f:
            byte_content = f.read()
    else:
        raise ValueError(f"{location} is neither a local file nor a valid url")

    mime = magic.Magic(mime=True)
    filetype = mime.from_buffer(byte_content)

    if "application/pdf" in filetype:
        text = extract_text(io.BytesIO(byte_content))
    elif "html" in filetype:
        soup = BeautifulSoup(byte_content.decode(), "lxml")
        text = soup.get_text()
    else:
        try:
            text = byte_content.decode()
        except UnicodeDecodeError:
            logger.exception(
                f"{location} does not appear to be a pdf document, an html document, or contain valid text."
            )
            raise

    return Document(text=text, doc_id=doc_id, location=location, digest=digest(text))
