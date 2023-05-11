import functools
import ast
import inspect
import importlib
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
            input_spec=self.input_spec,
            output_spec=output_spec,
            instructions=self.instructions,
            input=input_dict,
        )
        if CONFIG.api_key is None:
            raise RuntimeError("CAPABILITIES_API_KEY is not set")
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


# todo: directly assign this as an attribute of `llm`? Or make `llm` a class?
def llm_inline(regenerate=True, num_tries=3):

    @llm
    def complete_code(initial_codes: str, codes_before: str, codes_after: str) -> str:
        """
        Given the initial lines of a function, the codes before the function, and the codes after the function,
        follow the docstring to complete the function so that
        1. the initial lines of the code remain unchanged.
        2. it executes without error in the context.
        3. it fulfills the requirements set by its docstring in the context.

        Args:
            initial_codes: the first few lines of the function. Including the function signature and the docstring.
            codes_before: the codes before the function.
            codes_after: the codes after the function

        Returns:
            The completed function.
        """
        ...

    def wrapper(func: Callable[P, R]):
        if not regenerate:  # no-op if `regenerate==False`
            return func

        lines, start_lineno = inspect.getsourcelines(func)
        code_before, code_after = get_llm_inline_context(func)

        for _ in range(num_tries):  # Try `num_tries` times
            completed = complete_code("\n".join(lines[1:]), code_before, code_after)

            # Try to parse out the function body, but if unsuccessful, fall back to the original completion.
            processed = postprocess(completed, func.__name__)
            completed = processed if processed else completed

            new_decorator = f"@llm_inline(regenerate=False)"
            if input(f"The following code will be written to your file: \n{completed}\nContinue? (y/n)") == "y":
                with open(inspect.getfile(func), "w") as f:
                    f.write(f"{code_before}\n{new_decorator}\n{completed}\n{code_after}")

                # Reload the module and the functions

                # todo: reload the script itself could result in ModuleNotFoundError: spec not found for the module '__main__'. Trying an dirty way.
                try:
                    module = inspect.getmodule(func)
                    importlib.reload(module)
                    globals().update(vars(module))
                    return getattr(module, func.__name__)
                except:
                    exec(completed, globals())
                    print(globals()[func.__name__])
                    return globals()[func.__name__]


    return wrapper


def get_llm_inline_context(func) -> tuple[str, str]:
    source_path = inspect.getfile(func)
    lines, start_lineno = inspect.getsourcelines(func)
    end_lineno = start_lineno + len(lines)

    with open(source_path, "r") as f:
        file_content = f.read().split("\n")

    return "\n".join(file_content[:start_lineno - 1]), "\n".join(file_content[end_lineno:])


def postprocess(code: str, func_name: str) -> str:
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return ""

    completed_func = None
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.decorator_list = []
            completed_func = ast.unparse(node)
            return completed_func

    return ""

