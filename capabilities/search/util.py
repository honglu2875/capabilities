import heapq
import itertools
import math
from pathlib import Path
import sys
import tempfile
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    overload,
    TypeVar,
)
from hashlib import blake2b
from diskcache import Cache

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


cache = Cache(Path(tempfile.gettempdir()) / "capabilities" / "diskcache.db")


def fst(x: Tuple[U, Any]) -> U:
    return x[0]


def snd(x: Tuple[Any, V]) -> V:
    return x[1]


def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])


def partition(
    pred: Callable[[T], bool], iterable: Iterable[T]
) -> tuple[Iterable[T], Iterable[T]]:
    """Use a predicate to partition entries into false entries and true entries

    origin: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter.

    origin: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        return
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


IterMap = Callable[[Iterable[T]], Iterable[T]]


def partition_map(
    iterable: Iterable[T],
    pred: Callable[[T], bool],
    mapfalse: Optional[Callable[[Iterable[T]], Iterable[T]]] = None,
    maptrue: Optional[Callable[[Iterable[T]], Iterable[T]]] = None,
):
    fs, ts = partition(lambda x: pred(snd(x)), enumerate(iterable))

    def mmap(fn, xs):
        a, bs = itertools.tee(xs)
        return zip(map(fst, a), (fn(snd(b)) for b in bs))

    if mapfalse is not None:
        fs = mmap(mapfalse, fs)
    if maptrue is not None:
        ts = mmap(maptrue, ts)
    return heapq.merge(fs, ts, key=fst)


def argbatch(s: Sequence[int], n: int) -> list[range]:
    """Returns a list of ranges partitioning s such that the sums of the values of each of the given ranges is less than or equal to n."""
    o = []
    acc = 0
    a = 0
    for i, x in enumerate(s):
        if x > n:
            raise ValueError(f"s[{i}] = {x} is too big")
        acc += x
        if acc > n:
            o.append(range(a, i))
            acc = x
            a = i
    o.append(range(a, len(s)))
    return o


def argwindow(total_size: int, window_size: int, overlap=0.5):
    step_size = math.ceil(window_size * overlap)
    assert total_size > 0
    assert window_size > 0
    assert step_size > 0
    window_start = 0
    while True:
        window_end = min(window_start + window_size, total_size)
        assert window_end <= total_size
        yield window_start, window_end
        if window_end >= total_size:
            break
        window_start += step_size


from functools import cached_property
from typing import Callable, Generic, TypeVar


@overload
def unzip(s: Iterable[tuple[U, V]]) -> tuple[list[U], list[V]]:
    ...


@overload
def unzip(s: Iterable[tuple[U, ...]], n=2) -> tuple[list[U], ...]:
    ...


def unzip(s, n=2):  # type: ignore
    def g(i):
        def f(x):
            assert len(x) == n
            return x[i]

        return f

    return tuple(list(map(g(i), y)) for i, y in enumerate(itertools.tee(s, n)))


class Bijection(Generic[U, V]):
    _forward: dict[U, V]
    _inverse: dict[V, U]

    def __init__(self):
        self._forward = {}
        self._inverse = {}

    def __setitem__(self, key: U, value: V):
        if key in self._forward:
            raise ValueError("key already in bijection")
        if value in self._inverse:
            raise ValueError("value already in bijection")
        self._forward[key] = value
        self._inverse[value] = key

    def __contains__(self, key: U) -> bool:
        return key in self._forward

    def __len__(self):
        return len(self._forward)

    def __getitem__(self, key: U) -> V:
        return self._forward[key]

    def __delitem__(self, key: U):
        assert key in self._forward
        del self._inverse[self._forward[key]]
        del self._forward[key]

    def keys(self):
        return self._forward.keys()

    def values(self):
        return self._inverse.keys()

    @cached_property
    def inv(self: "Bijection[U, V]") -> "Bijection[V, U]":
        x = Bijection()
        x._forward = self._inverse
        x._inverse = self._forward
        return x

    def conj(self, f: Callable[[V], V]) -> Callable[[U], U]:
        return lambda u: self.inv[f(self[u])]  # type: ignore

    def __getstate__(self) -> dict:
        return self._forward

    def __setstate__(self, d: dict):
        assert isinstance(d, dict)
        self._forward = d
        self._inverse = {v: k for k, v in self._forward.items()}
        assert len(self._forward) == len(self._inverse), "not a bijection"


def digest(item: Union[str, bytes]):
    """ Computes the blake2b digest of a given item. """
    h = blake2b()
    if isinstance(item, str):
        item = item.encode("utf-8")
    h.update(item)
    return h.hexdigest()
