import numpy as np

from .types import VectorIndex


def topk(items: np.ndarray, k: int, axis=-1):
    """topk along given axis. If axis has dimension less than k, returns all items sorted.
    Args:
        items: array to search
        k: number of items to return
        axis: axis to search along

    Returns:
        - top k items
        - their indices in original array
    """
    # items : B, X => float
    top_ids = np.argpartition(items, -k, axis=axis)
    rest, top_ids = np.split(top_ids, [-k], axis=axis)
    # top_ids : B, K1 => X
    x = np.take_along_axis(items, top_ids, axis=axis)
    # x : B, K1 => float
    s = np.argsort(-x, axis=axis)
    # s : B, K2 => K1
    y = np.take_along_axis(x, s, axis=axis)
    i = np.take_along_axis(top_ids, s, axis=axis)
    # i : K2 => X
    return y, i


class SimpleVectorIndex(VectorIndex):
    """Simple in-mem vector index with cosine similarity."""

    def __init__(self):
        self.rows = np.empty((0, 0))

    def __len__(self):
        return len(self.rows)

    @property
    def dim(self):
        d = self.rows.shape[1]
        if d == 0:
            return None
        else:
            return d

    def add(self, arr):
        assert len(arr.shape) == 2, f"arr.shape={arr.shape}"
        norm = np.linalg.norm(arr, axis=1)
        if np.any(norm == 0):
            raise ValueError("vector with norm 0 detected")
        arr /= norm[:, np.newaxis]
        if len(self) == 0:
            self.rows = arr
        else:
            assert arr.shape[1] == self.rows.shape[1], "Dimensions do not match"
            self.rows = np.concatenate([self.rows, arr])

    def search(self, queries: np.ndarray, limit: int):
        batch_size, d = queries.shape
        if len(self) == 0:
            return (
                np.empty((batch_size, 0), dtype=np.float32),
                np.empty((batch_size, 0), dtype=np.int32),
            )
        assert d == self.dim, "Dimensions do not match"
        qnorm = np.linalg.norm(queries, axis=1)
        sim = np.einsum("ij,kj", queries / qnorm[:, np.newaxis], self.rows)
        return topk(sim, limit, axis=1)

    def __setstate__(self, state):
        self.rows = state["rows"]

    def __getstate__(self):
        return {"rows": self.rows}
