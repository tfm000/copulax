"""Integrity guard for the committed frozen-series corpus.

``copulax/tests/_r_reference/frozen_series_data.py`` records a SHA-256
for every series it commits, and the regenerator verifies every digest
at write time — but a tripwire that never runs protects nothing.  A
generated module of this size is a classic merge-conflict magnet: a
mangled conflict resolution or a hand edit (despite "DO NOT EDIT")
would silently become the new truth for every consumer of the corpus
and surface only as confusing downstream failures.  This module arms
the documented check on every suite run, so any byte-level corruption
of the committed data fails loudly, by series name, before a single
fit reads it.

Deliberately dependency-light: pure ``numpy`` + ``hashlib``, no jax, no
copulax model code and no fits — the whole corpus verifies in well
under a second.
"""

import hashlib

import numpy as np

from copulax.tests._r_reference.frozen_series_data import FROZEN_SERIES


def test_frozen_corpus_integrity():
    """Every committed series matches its own recorded provenance.

    For each ``FROZEN_SERIES`` entry: the array is 1-D float64
    (consumers downcast at the call site — a committed dtype change
    would alter every digest), the recorded ``n`` equals ``len(y)``,
    and the recorded SHA-256 equals the SHA-256 of the array's bytes —
    the exact check the corpus module's docstring documents, executed
    instead of quoted.
    """
    assert FROZEN_SERIES, "frozen corpus is empty"
    for name, entry in FROZEN_SERIES.items():
        y = np.asarray(entry["y"])
        provenance = entry["provenance"]
        assert y.dtype == np.float64, (name, y.dtype)
        assert y.ndim == 1, (name, y.shape)
        assert len(y) == provenance["n"], (name, len(y), provenance["n"])
        digest = hashlib.sha256(y.tobytes()).hexdigest()
        assert digest == provenance["sha256"], name
