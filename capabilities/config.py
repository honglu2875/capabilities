from dataclasses import dataclass
import os
import termcolor as tc


@dataclass
class Config:
    api_key: str


CONFIG = Config(api_key=os.environ.get("CAPABILITIES_API_KEY"))


def warn():
    import sys; out = sys.stdout
    msg_suffix = f"CAPABILITIES_API_KEY not set, get one here: {tc.colored('https://blazon.ai/signin', 'red')}"
    out.write(f" [   {tc.colored('warning', 'red', attrs=['bold'])}   ] " + msg_suffix); out.write("\n")

if CONFIG.api_key is None:
    warn()
