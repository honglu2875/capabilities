import dataclasses
import dacite
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Any, List, Union
import multiflow
from typing import Optional
import requests
import aiohttp
import asyncio
import time
from capabilities.config import CONFIG
from pydantic import BaseModel
from pydantic.main import ModelMetaclass


@dataclass
class CapabilityBase:
    ...


@dataclass
class DocumentQA(CapabilityBase):
    def __call__(self, document: str, query: str):
        print(f"[DocumentQA] running query against document with {len(document)} characters")
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.multi.dev/documentqa"
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
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
        print(f"[DocumentQA] running query against document with {len(document)} characters")
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.multi.dev/documentqa"
                if session is None:
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(ssl=False)
                    ) as session:
                        return await self.run_async(document=document, query=query, session=session)
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
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
    def __call__(self, document: str):
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.multi.dev/summarization"
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
                payload = {
                    "document": document,
                }
                print(f"[Summarize] running query against document with {len(document)} characters")
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
                url = "https://api.multi.dev/summarization"
                if session is None:
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(ssl=False)
                    ) as session:
                        return await self.run_async(document=document, session=session)
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
                payload = {
                    "document": document,
                }
                print(f"[Summarize] running query against document with {len(document)} characters")
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


@dataclass
class Ask(CapabilityBase):
    headers: Dict[Any, Any] = field(
        default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key}
    )
    url: str = "https://api.multi.dev/ask"

    def __call__(self, query: str):
        payload = dict(query=query)
        r = requests.post(self.url, headers=self.headers, json=payload)
        return r.json()

    async def run_async(self, query: str, session=None):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(
                    self.url, headers=self.headers, json=dict(query=query)
                ) as resp:
                    result = await resp.json()
                    return result
        else:
            async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                result = await resp.json()
                return result


@dataclass
class Sql(CapabilityBase):
    headers: Dict[Any, Any] = field(
        default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key}
    )
    url: str = "https://api.multi.dev/sql"

    def __call__(self, query: str, sql_schema: str, sql_variant: Optional[str] = "vanilla"):
        payload = dict(query=query, sql_schema=sql_schema, sql_type=sql_variant)
        r = requests.post(self.url, headers=self.headers, json=payload)
        return r.json()

    async def run_async(
        self, query: str, sql_schema: str, sql_variant: Optional[str] = "vanilla", session=None
    ):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(
                    self.url,
                    headers=self.headers,
                    json=dict(query=query, sql_schema=sql_schema, sql_type=sql_variant),
                ) as resp:
                    result = await resp.json()
                    return result
        else:
            async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                result = await resp.json()
                return result


@dataclass
class Search(CapabilityBase):
    """
    Run a web search, summarizing the top results.
    """

    headers: Dict[Any, Any] = field(
        default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key}
    )
    url: str = "https://api.multi.dev/search"

    def __call__(self, query: str):
        payload = dict(query=query)
        r = requests.post(self.url, headers=self.headers, json=payload)
        return r.json()

    async def run_async(self, query: str, session=None):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(
                    self.url, headers=self.headers, json=dict(query=query)
                ) as resp:
                    result = await resp.json()
                    return result
        else:
            async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                result = await resp.json()
                return result


def flatten_model(m: Union[ModelMetaclass, str, bool, float, int]):
    if hasattr(m, "__dict__"):
        if m.__dict__.get("_name") == "List":
            return [flatten_model(m.__args__[0])]
    if isinstance(m, ModelMetaclass) or is_dataclass(m):
        return {k: flatten_model(v) for k, v in m.__annotations__.items()}
    elif m == str:
        return "string"
    elif m == bool:
        return "bool"
    elif m == float:
        return "float"
    elif m == int:
        return "int"
    else:
        raise Exception(f"unsupported datatype={m}")


@dataclass
class Structured(CapabilityBase):
    """
    Run a web search, summarizing the top results.
    """

    headers: Dict[Any, Any] = field(
        default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key}
    )
    url: str = "https://api.multi.dev/structured"

    def __call__(
        self,
        input_spec: ModelMetaclass,
        output_spec: ModelMetaclass,
        instructions: str,
        input: BaseModel,
    ):
        payload = dict(
            input_spec=flatten_model(input_spec),
            output_spec=flatten_model(output_spec),
            instructions=instructions,
            input=dataclasses.asdict(input) if is_dataclass(input) else input.dict(),
        )
        print("PAYLOAD: ", payload)
        r = requests.post(self.url, headers=self.headers, json=payload)
        result = r.json()["output"]
        return (
            output_spec.parse_obj(result)
            if isinstance(output_spec, ModelMetaclass)
            else dacite.from_dict(output_spec, result)
        )

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
            input=dataclasses.asdict(input) if is_dataclass(input) else input.dict(),
            instructions=instructions,
        )
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(self.url, headers=self.headers, json=payload) as resp:
                    result = await resp.json()
                    return (
                        output_spec.parse_obj(result)
                        if isinstance(output_spec, ModelMetaclass)
                        else dacite.from_dict(output_spec, result)
                    )
        else:
            async with session.post(self.url, headers=self.headers, json=payload) as resp:
                result = await resp.json()
                return (
                    output_spec.parse_obj(result)
                    if isinstance(output_spec, ModelMetaclass)
                    else dacite.from_dict(output_spec, result)
                )


_CAPABILITIES_DIRECTORY = {
    "multi/summarize": Summarize(),
    "multi/document_qa": DocumentQA(),
    "multi/ask": Ask(),
    "multi/sql": Sql(),
    "multi/search": Search(),
    "multi/structured": Structured(),
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
            self._capability = _CAPABILITIES_DIRECTORY[self.uri]
        except KeyError as e:
            print(f"Capability lookup failed for uri={self.uri}.\nValid URIs are:")
            for k in _CAPABILITIES_DIRECTORY.keys():
                print(f"  {k}")
