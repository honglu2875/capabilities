from capabilities.search.util import argbatch, argwindow, unzip
from hypothesis import given
import hypothesis.strategies as st
import bisect


def test_argbatch():
    N = 15
    S = N
    l = list(range(N))
    b = argbatch(l, S)
    s = [l[x.start : x.stop] for x in b]
    r = [x for y in s for x in y]
    assert r == l
    assert all(sum(x) <= S for x in s)


@given(
    N=st.integers(min_value=1, max_value=1000),
    n=st.integers(min_value=1, max_value=1000),
    overlap=st.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_argwindow(N, n, overlap):
    windows = list(argwindow(N, n, overlap))
    for window_start, window_end in windows:
        assert N >= window_end > window_start >= 0
    starts, ends = unzip(windows)
    assert sorted(starts) == starts
    assert sorted(ends) == ends
    # assert each i belongs to at least one window.
    assert all(
        bisect.bisect_right(starts, i) >= bisect.bisect_left(ends, i) for i in range(N)
    )


@given(xs=st.from_type(list[tuple[int, int, int]]))
def test_unzip(xs):
    assert xs == list(zip(*unzip(xs, n=3)))
