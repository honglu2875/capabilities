import os
import dataclasses
import typing
import dacite
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Any, List, Type, TypeAlias, TypeVar, Union, Literal
from typing import Optional
import requests
import aiohttp
import asyncio
import time
from capabilities.config import CONFIG
from pydantic import BaseModel
from pydantic.fields import ModelField
from pydantic.main import ModelMetaclass
import logging

logger = logging.getLogger("capabilities")


@dataclass
class CapabilityBase:
    ...


@dataclass
class DocumentQA(CapabilityBase):
    """
    DocumentQA capability that sends the given query to DocumentQA service for answering based on the provided document.

    Attributes:
        None

    Methods:
        __call__(self, document: str, query: str) -> dict:
            Sends the given query to DocumentQA service for answering based on the provided document.
            Args:
                document: A string representing the input document.
                query: A string representing the query for DocumentQA.
            Returns:
                A dictionary containing the answer returned by the DocumentQA service.
            Raises:
                Exception: When the retries hit maximum (8) times and nothing was returned.

        async run_async(self, document: str, query: str, session=None) -> coroutine:
            Sends the given query to DocumentQA service for answering based on the provided document asynchronously.
            Args:
                document: A string representing the input document.
                query: A string representing the query for DocumentQA.
                session: An instance of `aiohttp.ClientSession`.
            Returns:
                A coroutine that resolves to a dictionary containing the answer returned
                by the DocumentQA service.
            Raises:
                Exception: When the retries hit maximum (8) times and nothing was returned.
    """

    def __call__(self, document: str, query: str):
        print(
            f"[DocumentQA] running query against document with {len(document)} characters"
        )
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.blazon.ai/blazon/documentqa"
                headers = {
                    "Content-type": "application/json",
                    "api-key": CONFIG.api_key,
                }
                payload = {
                    "document": document,
                    "query": query,
                }
                resp = requests.post(url=url, headers=headers, json=payload)
                return resp.json()
            except:
                sleep_duration = 2.0**count
                print(f"retrying after sleeping for {sleep_duration:.2f}")
                count += 1
                time.sleep(sleep_duration)
        raise Exception("[DocumentQA] failed after hitting max retries")

    async def run_async(self, document: str, query: str, session=None):
        print(
            f"[DocumentQA] running query against document with {len(document)} characters"
        )
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.blazon.ai/blazon/documentqa"
                if session is None:
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(ssl=False)
                    ) as session:
                        return await self.run_async(
                            document=document, query=query, session=session
                        )
                headers = {
                    "Content-type": "application/json",
                    "api-key": CONFIG.api_key,
                }
                payload = {
                    "document": document,
                    "query": query,
                }
                async with session.post(url, headers=headers, json=payload) as resp:
                    response = await resp.json()
                    try:
                        return response
                    except Exception as e:
                        print(f"caught exception={e}")
                        print(f"bad response={response}")
                        raise e
            except:
                sleep_duration = 2.0**count
                print(f"retrying after sleeping for {sleep_duration:.2f}")
                count += 1
                time.sleep(sleep_duration)
        raise Exception("[DocumentQA] failed after hitting max retries")


@dataclass
class Summarize(CapabilityBase):
    """
    Class for summarizing text using an API call to https://api.blazon.ai/blazon/summarize.

    Args:
        CapabilityBase (class): Base class for all capabilities.

    Methods:
        __call__(self, document: str) -> Dict[str, Any]:
            Method for summarizing `document`. Makes a POST request to the API and returns the JSON response.
            Retries up to 8 times with exponentially increasing sleep times before giving up.
            Args:
                document (str): The text to be summarized.
            Returns:
                Dict[str, Any]: A dictionary object representing the summary, with keys 'summary' (str) and 'score' (float).

        async run_async(self, document: str, session=None) -> Dict[str, Any]:
            Async method for summarizing `document`. Makes an async POST request to the API and returns the JSON response.
            Retries up to 8 times with exponentially increasing sleep times before giving up.
            Args:
                document (str): The text to be summarized.
                session (aiohttp.ClientSession, optional): An aiohttp client session. If not provided, a new one is created. Defaults to None.
            Returns:
                Dict[str, Any]: A dictionary object representing the summary, with keys 'summary' (str) and 'score' (float).
    """

    def __call__(self, document: str):
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.blazon.ai/blazon/summarize"
                headers = {
                    "Content-type": "application/json",
                    "api-key": CONFIG.api_key,
                }
                payload = {
                    "document": document,
                }
                print(
                    f"[Summarize] running query against document with {len(document)} characters"
                )
                resp = requests.post(url=url, headers=headers, json=payload)
                return resp.json()
            except:
                sleep_duration = 2.0**count
                print(f"retrying after sleeping for {sleep_duration:.2f}")
                count += 1
                time.sleep(sleep_duration)
        raise Exception("[Summarize] failed after hitting max retries")

    async def run_async(self, document: str, session=None):
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.blazon.ai/blazon/summarize"
                if session is None:
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(ssl=False)
                    ) as session:
                        return await self.run_async(document=document, session=session)
                headers = {
                    "Content-type": "application/json",
                    "api-key": CONFIG.api_key,
                }
                payload = {
                    "document": document,
                }
                print(
                    f"[Summarize] running query against document with {len(document)} characters"
                )
                async with session.post(url, headers=headers, json=payload) as resp:
                    response = await resp.json()
                    try:
                        return response
                    except Exception as e:
                        print(f"caught exception={e}")
                        print(f"bad response={response}")
                        raise e
            except:
                sleep_duration = 2.0**count
                print(f"retrying after sleeping for {sleep_duration:.2f}")
                count += 1
                time.sleep(sleep_duration)
        raise Exception("[Summarize] failed after hitting max retries")


StructuredSchema: TypeAlias = Any


def flatten_model(m: Type, path: list[str] = []) -> StructuredSchema:
    """Converts the given type m to a structured schema.

    Args:
        * m (Type): The type to be converted.
        * path (list[str], optional): The path to the current type,
          used for more helpful diagnostic messages saying where
          the conversion failed. Defaults to [].
    """
    orig = typing.get_origin(m)
    if m == list or orig == list:
        args = typing.get_args(m)
        if len(args) != 1:
            p = ".".join(path)
            raise TypeError(
                f"can't deduce list type arg for {m} at {p}, please provide a type annotation to list.",
                path,
            )
        t = flatten_model(args[0])
        return [t]
    elif isinstance(m, ModelMetaclass):
        fields: dict[str, ModelField] = m.__fields__ # type: ignore
        return {
            k: flatten_model(f.annotation, path=path + [k]) for k, f in fields.items()
        }
    elif is_dataclass(m):
        return {
            f.name: flatten_model(f.type, path=path + [f.name])
            for f in dataclasses.fields(m)
        }
    elif m == str:
        return "string"
    elif m == bool:
        return "bool"
    elif m == float:
        return "float"
    elif m == int:
        return "int"
    else:
        p = ".".join(path)
        raise TypeError(
            f"unsupported datatype={m} at {p}\nPlease make sure that all fields are annotated with a bool, int, float, str or a dataclass, list, or pydantic BaseModel. If you need other types supported, let us know!"
        )


def to_dict(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif is_dataclass(obj):
        return dataclasses.asdict(obj)
    elif isinstance(obj, list):
        return [to_dict(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (str, bool, float, int)):
        return obj
    else:
        raise TypeError(f"unsupported datatype={obj}")


T = TypeVar("T")


def of_dict(t: Type[T], d: Any) -> T:
    if isinstance(t, ModelMetaclass):
        assert isinstance(d, dict)
        return t.parse_obj(d)
    elif is_dataclass(t):
        assert isinstance(d, dict)
        return dacite.from_dict(t, d)  # type: ignore
    elif t == list or typing.get_origin(t) == list:
        assert isinstance(d, list)
        args = typing.get_args(t)
        if len(args)!= 1:
            return d # type: ignore
        (arg,) = args
        return [of_dict(arg, o) for o in d]  # type: ignore
    elif t == dict:
        (kt, vt) = typing.get_args(t)
        assert isinstance(d, dict)
        assert kt in [str, int]
        return {of_dict(kt, k): of_dict(vt, v) for k, v in d.items()}  # type: ignore
    elif t in [str, int, float, bool]:
        assert isinstance(d, t)
        return d
    else:
        raise TypeError(f"unsupported datatype={t}")


@dataclass
class Structured(CapabilityBase):
    """
    `Structured` class allows making requests to the multi API for structured tasks. The class extends CapabilityBase which provides required functionality to interact with multi API. Structured tasks are tasks with specific input and output specs with a natural language instruction.

    Attributes:
        headers (Dict[Any, Any]): HTTP headers to use in requests. It includes the Content-type and API Key parameters.
        url (str): API endpoint URL

    Methods:
        __call__(self, input_spec: ModelMetaclass, output_spec: ModelMetaclass, instructions: str, input: BaseModel) -> Union[output_spec, BaseModel]: Calls the API by sending a payload within a request object. Returns output_spec object if output_spec is ModelMetaclass or if it is an instance of a BaseModel.

        async run_async(self, input_spec: ModelMetaclass, output_spec: ModelMetaclass, instructions: str, input: BaseModel, session=None) -> Union[output_spec, BaseModel]: Calls the API asynchronously. Returns output_spec object if output_spec is ModelMetaclass or if it is an instance of a BaseModel.
    """

    headers: dict = field(
        default_factory=lambda: {
            "Content-type": "application/json",
            "api-key": CONFIG.api_key,
        }
    )
    url: str = "https://api.blazon.ai/blazon/structured"

    def __call__(
        self,
        input_spec: ModelMetaclass,
        output_spec: ModelMetaclass,
        instructions: str,
        input: Any,
    ):
        payload = dict(
            input_spec=flatten_model(input_spec),
            output_spec=flatten_model(output_spec),
            instructions=instructions,
            input=to_dict(input),
        )
        r = requests.post(self.url, headers=self.headers, json=payload)
        r.raise_for_status()
        logger.debug("R: ", r)
        result = r.json()["output"]
        return of_dict(output_spec, result)

    async def run_async(
        self,
        input_spec: ModelMetaclass,
        output_spec: ModelMetaclass,
        instructions: str,
        input: BaseModel,
        session=None,
    ):
        payload = dict(
            input_spec=flatten_model(input_spec),
            output_spec=flatten_model(output_spec),
            input=to_dict(input),
            instructions=instructions,
        )
        if session is None:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False)
            ) as session:
                async with session.post(
                    self.url, headers=self.headers, json=payload
                ) as resp:
                    result = await resp.json()
                    result = result["output"]
                    return of_dict(output_spec, result)
        else:
            async with session.post(
                self.url, headers=self.headers, json=payload
            ) as resp:
                result = (await resp.json())["output"]
                return of_dict(output_spec, result)


_CAPABILITIES = {
    "multi/structured": Structured(),
    "multi/document_qa": DocumentQA(),
    "multi/summarize": Summarize(),
    "blazon/structured": Structured(),
    "blazon/document_qa": DocumentQA(),
    "blazon/summarize": Summarize(),
}


@dataclass
class Capability(CapabilityBase):
    uri: str
    _capability: Optional[CapabilityBase] = None

    def __call__(self, *args, **kwargs):
        return self._capability(*args, **kwargs)

    async def run_async(self, *args, **kwargs):
        return await self._capability.run_async(*args, **kwargs)

    def __post_init__(self):
        try:
            self._capability = _CAPABILITIES[self.uri]
        except KeyError as e:
            print(f"Capability lookup failed for uri={self.uri}.\nValid URIs are:")
            for k in _CAPABILITIES.keys():
                print(f"  {k}")
