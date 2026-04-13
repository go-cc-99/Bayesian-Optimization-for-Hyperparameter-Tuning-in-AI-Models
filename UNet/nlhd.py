#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlhd_simulation.py

Reusable Python implementations of:

- random_lhs(n, k): basic Latin hypercube sampling in [0,1]^k
- nlhd(S, k): nested Latin hypercube design (translation of R code)
- fun1(x): test function log(1/x1 + 1/x2)
- simulate_nlhd(): run the 2000-repetition simulation (optional)

This file can be imported directly:

    from nlhd_simulation import nlhd, random_lhs, fun1

"""

import numpy as np


# ----------------------------------------------------------------------
# Latin Hypercube Sampling in [0,1]^k
# ----------------------------------------------------------------------
def random_lhs(n: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate an n × k Latin Hypercube Sample in [0,1]^k."""
    if rng is None:
        rng = np.random.default_rng()

    x = np.zeros((n, k))
    for j in range(k):
        # divide [0,1) into n segments and pick 1 per segment
        u = (rng.random(n) + np.arange(n)) / n
        rng.shuffle(u)
        x[:, j] = u
    return x


# ----------------------------------------------------------------------
# Nested Latin Hypercube Design (Translation of the R version)
# ----------------------------------------------------------------------
def nlhd(S, k: int, rng: np.random.Generator | None = None):
    """
    Generate Nested Latin Hypercube Design.

    Parameters
    ----------
    S : list of ints
        Layer sizes, e.g. [10,2].
    k : int
        Dimension.
    rng : np.random.Generator, optional.

    Returns
    -------
    dict with:
        - "xmat": (n,k) array, scaled design points in (0,1)
        - "pmat": (n,k) array, integer positions
    """
    if rng is None:
        rng = np.random.default_rng()

    S = np.asarray(S, dtype=int)
    k = int(k)
    u = len(S)
    n = int(np.prod(S))

    pmat = np.zeros((n, k), dtype=int)
    xmat = np.zeros((n, k), dtype=float)

    for p in range(k):
        D = np.zeros(n, dtype=int)
        oldm = 0

        for i in range(u):
            newm = int(np.prod(S[: i + 1]))
            t = n // newm

            if i > 0:
                oldm = int(np.prod(S[:i]))
                C = np.ceil(D[:oldm] / t).astype(int)
            else:
                C = np.array([], dtype=int)

            all_indices = np.arange(1, newm + 1)
            PI = np.setdiff1d(all_indices, C)
            PI = rng.choice(PI, size=(newm - oldm), replace=False)

            if t == 1:
                D[oldm:newm] = rng.permutation(PI)
            else:
                for j in range(newm - oldm):
                    start = (PI[j] - 1) * t + 1
                    end = PI[j] * t
                    D[oldm + j] = rng.integers(start, end + 1)

        U = rng.random(n)
        pmat[:, p] = D
        xmat[:, p] = (D - U) / n  # same scaling as R: (D - U)/n

    return {"xmat": xmat, "pmat": pmat}

def check_lhd(pmat):
    n, k = pmat.shape
    for p in range(k):
        col = pmat[:, p]
        if set(col) != set(range(1, n + 1)):
            print(f"Column {p} is not a permutation of 1..{n}")
            return False
    return True
