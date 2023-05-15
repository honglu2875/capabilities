from dataclasses import dataclass
import os
from typing import Optional
import termcolor as tc
import pydantic
import sys


class Config(pydantic.BaseSettings):
    api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "CAPABILITIES_"

    def __post_init__(self):
        if self.api_key is None:
            out = sys.stdout
            msg_suffix = f"CAPABILITIES_API_KEY not set, get one here: {tc.colored('https://blazon.ai/signin', 'red')}"
            out.write(f" [   {tc.colored('warning', 'red', attrs=['bold'])}   ] " + msg_suffix)
            out.write("\n")


CONFIG = Config()
