import os
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
        print(f"[DocumentQA] running query against document with {len(document)} characters")
        patience = 8
        count = 0
        while count < patience:
            try:
                url = "https://api.multi.dev/blazon/documentqa"
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
                payload = {
                    "document": document,
                    "query": query,
                }
                resp = requests.post(url=url, headers=headers, json=payload)
                import curl

                curl.parse(resp)
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
                url = "https://api.multi.dev/blazon/documentqa"
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
    """
    Class for summarizing text using an API call to https://api.multi.dev/blazon/summarize.

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
                url = "https://api.multi.dev/blazon/summarize"
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
                url = "https://api.multi.dev/blazon/summarize"
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
class Sql(CapabilityBase):
    headers: Dict[Any, Any] = field(
        default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key}
    )
    url: str = "https://api.multi.dev/blazon/sql"

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
    url: str = "https://api.multi.dev/blazon/search"

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
class WebContent(CapabilityBase):
    def __call__(self, url):
        endpoint_url = (
            f"https://chrome.browserless.io/content?token={os.environ.get('BROWSERLESS_API_KEY')}"
        )

        payload = {"url": url}

        headers = {"Content-Type": "application/json", "Cache-Control": "no-cache"}
        response = requests.post(endpoint_url, json=payload, headers=headers)
        return Document(response._content.decode("utf-8")).content()

    async def run_async(self, url, session=None):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                return await self.run_async(url, session=session)
        else:
            payload = {"url": url}
            endpoint_url = f"https://chrome.browserless.io/content?token={os.environ.get('BROWSERLESS_API_KEY')}"
            headers = {"Content-Type": "application/json", "Cache-Control": "no-cache"}
            async with session.post(endpoint_url, headers=headers, json=payload) as resp:
                result = (await resp.content.read()).decode("utf-8")
                return Document(result).summary()


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

    headers: Dict[Any, Any] = field(
        default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key}
    )
    url: str = "https://api.multi.dev/blazon/structured"
    # url: str = "https://multi-api-3lt5g6vshq-uc.a.run.app/structured"

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
                    result = result["output"]
                    return (
                        output_spec.parse_obj(result)
                        if isinstance(output_spec, ModelMetaclass)
                        else dacite.from_dict(output_spec, result)
                    )
        else:
            async with session.post(self.url, headers=self.headers, json=payload) as resp:
                result = (await resp.json())["output"]
                return (
                    output_spec.parse_obj(result)
                    if isinstance(output_spec, ModelMetaclass)
                    else dacite.from_dict(output_spec, result)
                )


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
