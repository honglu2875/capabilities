from capabilities.search import SimpleVectorIndex, topk
import numpy as np


def test_topk1():
    x = np.arange(6)
    d, i = topk(x, k=2)
    assert d.shape == (2,)
    assert i.shape == (2,)
    assert np.array_equal(np.take_along_axis(x, i, axis=0), d)
    assert d.tolist() == [5, 4]


def test_topk2():
    x = np.stack([np.arange(6), -np.arange(6), np.arange(6)[::-1]], axis=0)
    assert x.shape == (3, 6)
    d, i = topk(x, k=2, axis=1)
    assert d.shape == (3, 2)
    assert i.shape == (3, 2)
    d_expected = np.array([[5, 4], [0, -1], [5, 4]])
    np.testing.assert_array_equal(d, d_expected)
    i_expected = np.array([[5, 4], [0, 1], [0, 1]])
    np.testing.assert_array_equal(i, i_expected)


def test_simple_vector_index():
    rows = np.eye(10)
    index = SimpleVectorIndex()
    assert index.dim == None
    index.add(rows)
    assert index.dim == 10
    sim, i = index.search(rows, limit=1)
    assert sim.shape == (10, 1)
    np.testing.assert_array_equal(sim, np.ones((10, 1)))
    np.testing.assert_array_equal(i[:, 0], np.arange(10))
    index.add(np.ones((1, 10)))
    sim, i = index.search(rows, limit=2)
    assert sim.shape == (10, 2)
    assert np.all(i[:, 1] == 10)
    np.testing.assert_almost_equal(sim[0, 1], 1.0 / np.sqrt(10))
