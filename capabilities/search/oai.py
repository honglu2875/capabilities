""" Open AI API client. """

from datetime import datetime
import json
from typing import Literal, Optional, Union
import numpy as np
from pydantic import BaseModel, BaseSettings, Field, SecretStr
import tiktoken
from .types import EmbeddingModel
from .util import argbatch, cache
from rich.progress import track
import requests

""" A client for the OpenAI API. """


class OpenAISettings(BaseSettings):
    api_key: SecretStr
    api_url: str = "https://api.openai.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "OPENAI_"

    def create_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
        }
        return headers

    def post(self, path: str, **kwargs):
        url = f"{self.api_url}{path}"
        return requests.post(url, headers=self.create_headers(), **kwargs)


class Message(BaseModel):
    """OpenAI chat message.

    ref: https://platform.openai.com/docs/guides/chat
    """

    role: Literal["system", "user", "assistant"]
    content: str

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)


class CompletionRequest(BaseModel):
    """Request body for OpenAI completion API.

    ref: https://platform.openai.com/docs/api-reference/chat/create
    """

    model: str = Field("gpt-3.5-turbo")
    """ ID of the model to use.
    See [model compatibility page](model endpoint compatibility).
    As of 2022-04-06 these are
    OpenAI: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
    Third-party values: gpt4all
    """
    messages: list[Message]


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "content_filter"]]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = Field(default=None)
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: datetime
    choices: list[Choice]
    usage: Optional[Usage]  # [todo] not optional?


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = Field(default="text-embedding-ada-002")
    user: Optional[str] = Field(default=None)


class EmbeddingObject(BaseModel):
    object: Literal["embedding"]
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"]
    data: list[EmbeddingObject]
    model: str = Field(default="text-embedding-ada-002")
    usage: Usage


# [todo] throttling, retries
@cache.memoize()
def embeddings(params: EmbeddingRequest) -> EmbeddingResponse:
    resp = OpenAISettings().post("/v1/embeddings", json=params.dict(exclude_none=True))  # type: ignore
    j = resp.json()
    if resp.status_code // 100 == 4:
        raise RuntimeError(f"{resp.status_code}: {json.dumps(j, indent=2)}")
    resp.raise_for_status()
    r = EmbeddingResponse.parse_obj(j)
    return r


@cache.memoize()
def chat_completions(params: CompletionRequest) -> CompletionResponse:
    resp = OpenAISettings().post("/v1/chat/completions", json=params.dict(exclude_none=True))  # type: ignore
    resp.raise_for_status()
    j = resp.json()
    r = CompletionResponse.parse_obj(j)
    return r


class OpenAIEmbeddingModel(EmbeddingModel):
    model_name = "text-embedding-ada-002"
    tokenizer_name = "cl100k_base"

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding(self.tokenizer_name)

    def __setstate__(self, state):
        self.__init__()

    def __getstate__(self):
        return {}

    @property
    def max_tokens_per_item(self) -> int:
        return 8191

    @property
    def dim(self) -> int:
        return 1536

    def count_tokens(self, text: str):
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def tokenize(self, text: str):
        enc = self.tokenizer.encode(text)
        return enc

    def detokenize(self, tokens: list[int]):
        return self.tokenizer.decode(tokens)

    def get_token_offsets(self, text: str):
        enc = self.tokenizer.encode(text)
        # this isn't technically correct, but close enough
        # generally ends[-1] >= len(text) meaning we underestimate token density.
        # it's tricky to get the true offsets without having support from tokenizer.
        ls = np.array([len(self.tokenizer.decode([x])) for x in enc])
        ends = np.cumsum(ls)
        starts = np.concatenate([[0], ends[:-1]])
        assert ends[-1] >= len(text)
        return starts, ends

    def encode(self, texts: list[str]):
        lengths = [self.count_tokens(text) for text in texts]
        N = self.max_tokens_per_item
        for i, l in enumerate(lengths):
            if l > N:
                raise ValueError(
                    f"{i}th text {texts[i]} is too long, please chunk it first"
                )
        batches = argbatch(lengths, N)
        es = []
        its = track(batches) if len(batches) > 100 else batches
        for batch in its:
            ts = texts[batch.start : batch.stop]
            responses = embeddings(EmbeddingRequest(model=self.model_name, input=ts))
            new_es = np.array([r.embedding for r in responses.data])
            assert len(new_es) == len(ts)
            es.append(new_es)
        es = np.concatenate(es, axis=0)
        assert len(es) == len(texts)
        return es
