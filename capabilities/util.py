import time
import aiohttp
import fire
import os
import requests
import openai
import random
import tiktoken

ENCODER = tiktoken.get_encoding("p50k_base")

def tokenize(contents):
    return ENCODER.encode(contents)

def detokenize(tokens):
    return ENCODER.decode(tokens)

def get_chunks_tokenized(contents: str, max_len: int = 256, stride=None):
    if stride is None:
        stride = max_len
    tokens = tokenize(contents)
    return [tokens[stride * i : stride * i + max_len] for i in range(len(tokens) // stride + 1)]

def get_tokenized_length(contents: str):
    return len(tokenize(contents))

def get_chunks_core(contents, max_len, stride=None):
    return [detokenize(tks) for tks in get_chunks_tokenized(contents, max_len, stride)]

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (KeyError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                print(f"[retry_with_exponential_backoff] retrying, num_retries={num_retries} max_retries={max_retries}")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def retry_with_exponential_backoff_async(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    async def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return await func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                print(f"[retry_with_exponential_backoff] retrying, num_retries={num_retries} max_retries={max_retries}")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff_async
async def text_embed_async(prompt, session=None):
    assert session is not None # lol
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY_EMBED')}",
    }
    payload = {
        "input": prompt,
        "model": "text-embedding-ada-002"
    }

    async with session.post(
        "https://api.openai.com/v1/embeddings", headers=headers, json=payload
    ) as resp:
        result = await resp.json()

        try:
            return result["data"][0]["embedding"]
        except KeyError as e:
            print("[text_generate_async] BAD RESPONSE: ", result)
            print(f"[text_generate_async] exception={e}")
            print("[text_generate_async] BAD PROMPT: ", prompt)
            raise e

@retry_with_exponential_backoff
def text_embed(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY_EMBED')}",
    }
    payload = {
        "input": prompt,
        "model": "text-embedding-ada-002"
    }
    r = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=payload)
    return r.json()["data"][0]["embedding"]
