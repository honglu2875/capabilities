from dataclasses import dataclass
import os


@dataclass
class Config:
    api_key: str


CONFIG = Config(api_key=os.environ.get("CAPABILITIES_API_KEY"))

if CONFIG.api_key is None:
    print("Warning: `CAPABILITIES_API_KEY` is not set")
