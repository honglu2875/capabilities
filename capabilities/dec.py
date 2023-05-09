import functools
import inspect
from typing import Callable, Generic, ParamSpec, TypeVar, overload
import warnings

import requests
from capabilities.config import CONFIG

from capabilities.core import (
    StructuredSchema,
    flatten_model,
    of_dict,
    to_dict,
)

P = ParamSpec("P")
R = TypeVar("R")


def _flatten_param(param: inspect.Parameter, path=[]) -> StructuredSchema:
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
        t = param.annotation
        assert isinstance(t, type)
        return flatten_model(list[t])
    elif param.kind == inspect.Parameter.VAR_KEYWORD:
        raise NotImplementedError("**kwargs arguments are not implemented yet, sorry")
        t = param.annotation
        assert isinstance(t, type)
        return flatten_model(dict[str, t])
    else:
        return flatten_model(param.annotation)


class AiFunction(Generic[P, R]):
    __wrapped__: Callable[P, R]

    def __init__(self, func, *, instructions=None, **kwargs):
        functools.update_wrapper(self, func)
        if instructions is not None:
            self.instructions = instructions
        else:
            # get instructions from docstring
            self.instructions = inspect.getdoc(func)
            if self.instructions is None:
                warnings.warn(
                    "using AiFunction without instructions. Please add a docstring to the decorated function."
                )
                self.instructions = (
                    "Please produce the given output from the given input."
                )
        self.signature = inspect.signature(func)
        self.input_spec = {
            k: _flatten_param(p, path=[k]) for k, p in self.signature.parameters.items()
        }
        self.output_spec = flatten_model(self.signature.return_annotation)

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        binding = self.signature.bind(*args, **kwargs)
        binding.apply_defaults()
        input_dict = {k: to_dict(v) for k, v in binding.arguments.items()}
        # [todo](ed) currently endpoint can't handle having root spec not be a dictionary.
        wrap_output = not isinstance(self.output_spec, dict)
        if wrap_output:
            output_spec = {"output": self.output_spec}
        else:
            output_spec = self.output_spec
        payload = dict(
            input_spec = self.input_spec,
            output_spec = output_spec,
            instructions = self.instructions,
            input = input_dict
        )
        resp = requests.post(
            "https://api.blazon.ai/blazon/structured",
            headers={
                "Content-Type": "application/json",
                "api-key": CONFIG.api_key,
            },
            json=payload,
        )
        resp.raise_for_status()
        result_dict = resp.json()["output"]
        if wrap_output:
            result_dict = result_dict["output"]
        result = of_dict(self.signature.return_annotation, result_dict)
        return result


@overload
def llm(*, instructions=None) -> Callable[[Callable[P, R]], AiFunction[P, R]]:
    ...


@overload
def llm(f: Callable[P, R]) -> AiFunction[P, R]:
    """Structured synthesis."""
    ...


def llm(*args, **kwargs):  # type: ignore
    def decorator(func):
        item = AiFunction(func)
        return item

    if callable(args[0]):
        return decorator(args[0])
    else:
        return functools.partial(AiFunction, *args, **kwargs)
