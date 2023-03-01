from dataclasses import dataclass, field
from typing import Dict, Any, List
import multiflow
from typing import Optional
import requests
import aiohttp
import asyncio
import time
from capabilities.config import CONFIG


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
                resp = requests.post(
                    url=url,
                    headers=headers,
                    json=payload
                )
                return resp.json()["answer"]
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
                        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                            return await self.run_async(document=document, query=query, session=session)
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
                payload = {
                    "document": document,
                    "query": query,
                }
                async with session.post(url, headers=headers, json=payload) as resp:
                    response = await resp.json()
                    try:
                        return response["answer"]
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
                resp = requests.post(
                    url=url,
                    headers=headers,
                    json=payload
                )
                return resp.json()["summary"]
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
                        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                            return await self.run_async(document=document, session=session)
                headers = {"Content-type": "application/json", "api-key": CONFIG.api_key}
                payload = {
                    "document": document,
                }
                print(f"[Summarize] running query against document with {len(document)} characters")
                async with session.post(url, headers=headers, json=payload) as resp:
                    response = await resp.json()
                    try:
                        return response["summary"]
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
    headers: Dict[Any, Any] = field(default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key})
    url: str = "https://api.multi.dev/ask"

    def __call__(self, query: str):
        payload = dict(query=query)
        r = requests.post(
        self.url, headers=self.headers, json=payload
        )
        return r.json()["answer"]

    async def run_async(self, query: str, session=None):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                    result = await resp.json()
                    return result["answer"]
        else:
            async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                result = await resp.json()
                return result["answer"]

@dataclass
class Sql(CapabilityBase):
    headers: Dict[Any, Any] = field(default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key})
    url: str = "https://api.multi.dev/sql"

    def __call__(self, query: str, sql_schema: str, sql_variant: Optional[str] = "vanilla"):
        payload = dict(query=query, sql_schema=sql_schema, sql_type=sql_variant)
        r = requests.post(self.url, headers=self.headers, json=payload)
        return r.json()["sql_query"]

    async def run_async(self, query: str, sql_schema: str, sql_variant: Optional[str] = "vanilla", session=None):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(self.url, headers=self.headers, json=dict(query=query, sql_schema=sql_schema, sql_type=sql_variant)) as resp:
                    result = await resp.json()
                    return result["sql_query"]
        else:
            async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                result = await resp.json()
                return result["sql_query"]

@dataclass
class Search(CapabilityBase):
    """
    Run a web search, summarizing the top results.
    """
    headers: Dict[Any, Any] = field(default_factory=lambda: {"Content-type": "application/json", "api-key": CONFIG.api_key})
    url: str = "https://api.multi.dev/search"

    def process_result(self, result: str):
        answer = result.split("--- sources:")[0].split("--- answer:")[1].lstrip().rstrip()
        sources = result.split("--- sources:")[1].lstrip().rstrip().split("\n")
        return dict(
            answer=answer,
            sources=sources,
        )
        

    def __call__(self, query: str):
        payload = dict(query=query)
        r = requests.post(
        self.url, headers=self.headers, json=payload
        )
        return self.process_result(r.json()["result"])

    async def run_async(self, query: str, session=None):
        if session is None:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                    result = await resp.json()
                    return self.process_result(result["result"])
        else:
            async with session.post(self.url, headers=self.headers, json=dict(query=query)) as resp:
                result = await resp.json()
                return self.process_result(result["result"])


_CAPABILITIES_DIRECTORY = {
    "multi/summarize": Summarize(),
    "multi/document_qa": DocumentQA(),
    "multi/ask": Ask(),
    "multi/sql": Sql(),
    "multi/search": Search(),
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

if __name__ == "__main__":
    c = Capability("multi/search")
    print(c("what are the seven wonders of the world?"))
