"""MinHash: a short signature whose agreement rate *is* the Jaccard similarity.

Take two sets and hash every element. Ask: which element got the smallest hash? Because the hash
is random-looking, every element of the union is equally likely to be the minimum, so the chance
that both sets agree on the minimum is exactly the chance that this element is in the
intersection::

    P(min_h(A) == min_h(B)) = |A n B| / |A u B| = J(A, B)

One hash gives one coin flip of evidence. Repeat with ``k`` independent hashes and the fraction
of positions where the signatures agree estimates J with standard error ``sqrt(J(1-J)/k)``:
128 permutations put that around 4 percentage points, from 128 numbers per document however long
the documents are (Broder, 1997).

Each permutation here is ``splitmix64(h(x) XOR salt_i)`` for a random 64-bit salt. Both steps are
bijections of the 64-bit space, so each is a genuine permutation, and SplitMix64's avalanche makes
the k of them behave independently - which is the property the ``sqrt(J(1-J)/k)`` error rests on.
(The textbook ``(a x + b) mod p`` family is only pairwise independent; with the small multipliers
needed to avoid 64-bit overflow it degenerates into near-monotone maps that all pick the same
minimum, which collapses the effective number of permutations and inflates the error.)
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Sequence

import numpy as np

from src.core.hashes import hash_many, splitmix64_array

MAX_HASH = (1 << 64) - 1

_WORD_RE = re.compile(r"\w+")


def shingles(text: str, size: int = 5, kind: str = "word") -> set[str]:
    """Split text into overlapping k-shingles, the set MinHash actually compares.

    ``kind="word"`` gives k-word windows (robust to formatting); ``kind="char"`` gives k-character
    windows (robust to word-level edits, and what near-duplicate detectors usually use on short
    text).
    """
    if size < 1:
        raise ValueError("size must be positive")
    if kind == "word":
        tokens = _WORD_RE.findall(text.lower())
        if len(tokens) < size:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}
    if kind == "char":
        cleaned = " ".join(text.lower().split())
        if len(cleaned) < size:
            return {cleaned} if cleaned else set()
        return {cleaned[i : i + size] for i in range(len(cleaned) - size + 1)}
    raise ValueError("kind must be 'word' or 'char'")


def jaccard(set_a: Collection[object], set_b: Collection[object]) -> float:
    """Exact Jaccard similarity - the truth every estimate here is scored against."""
    left, right = set(set_a), set(set_b)
    union = len(left | right)
    if union == 0:
        return 1.0
    return len(left & right) / union


class MinHasher:
    """Turns any set into a fixed-length signature that preserves Jaccard similarity.

    >>> hasher = MinHasher(num_perm=128, seed=0)
    >>> a = hasher.signature({"the", "quick", "brown", "fox"})
    >>> b = hasher.signature({"the", "quick", "brown", "cat"})
    >>> 0.3 < hasher.estimated_jaccard(a, b) < 0.9
    True
    """

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.seed = seed
        rng = np.random.default_rng(seed)
        # One random 64-bit salt per permutation; XOR then mix gives k independent bijections.
        high = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
        low = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
        self._salts = (high << np.uint64(32)) | low

    def __repr__(self) -> str:
        return f"MinHasher(num_perm={self.num_perm}, seed={self.seed})"

    def signature(self, items: Iterable[object], chunk_size: int = 4096) -> np.ndarray:
        """Signature of a set: the minimum of each permutation over its elements."""
        elements = list(items)
        best = np.full(self.num_perm, MAX_HASH, dtype=np.uint64)
        for start in range(0, len(elements), chunk_size):
            chunk = elements[start : start + chunk_size]
            base = hash_many(chunk, self.seed)
            permuted = splitmix64_array(base.reshape(-1, 1) ^ self._salts.reshape(1, -1))
            best = np.minimum(best, permuted.min(axis=0))
        return best

    def signature_matrix(self, sets: Sequence[Iterable[object]]) -> np.ndarray:
        """Signatures for many sets at once, shape ``(n_sets, num_perm)``."""
        return np.vstack([self.signature(one) for one in sets]) if sets else np.zeros((0, 0))

    def estimated_jaccard(self, signature_a: np.ndarray, signature_b: np.ndarray) -> float:
        """Fraction of positions where two signatures agree - the estimate of J."""
        if signature_a.shape != signature_b.shape:
            raise ValueError("signatures must have the same length")
        return float(np.mean(signature_a == signature_b))

    def error_band(self, similarity: float, sigmas: float = 3.0) -> float:
        """Sampling error of the estimate at a given true similarity: ``sqrt(J(1-J)/k)``."""
        return sigmas * math.sqrt(max(similarity * (1.0 - similarity), 1e-12) / self.num_perm)

    def memory_bytes(self) -> int:
        """Bytes one signature occupies - the same for a tweet and for a novel."""
        return int(self.num_perm * np.dtype(np.uint64).itemsize)
