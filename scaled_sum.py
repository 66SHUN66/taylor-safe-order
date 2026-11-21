#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scaled_sum.py
=============

The accumulator that makes partial sums of a Taylor series overflow-proof.

A double holds |x| <= 1.8e308.  Terms of a Taylor series routinely pass
*outside* that window on the way to a perfectly representable final value
(e.g. the e**x series peaks at e**x/sqrt(2*pi*x), which is ~1e306 for x = 700).
So instead of storing the running sum as one float, store it as

        S = m * 2**e ,      |m| < 2 ,    e an ordinary int (may exceed 1024)

Only the final value() has to live in the double range; the accumulator
itself never leaves it.  Each addition aligns the two exponents, exactly the
way hand-written floating point arithmetic works, so the bits that fall off
the bottom are bits the final double could not have kept anyway.

What this does NOT do
---------------------
Scaling fixes overflow.  It does not fix cancellation: if terms reach 1e300
and the answer is 1e-300, ~600 decimal digits are lost no matter how the sum
is stored.  Use an identity instead (e**-x = 1/e**x, log1p, ...).
"""

import math

__all__ = ["ScaledSum"]


class ScaledSum:
    """Partial sum stored as  m * 2**e  with |m| < 2.

    Example
    -------
        acc = ScaledSum()
        for t in terms:
            acc.add(t)
        value = acc.value()      # raises OverflowError only if |S| > DBL_MAX
    """

    __slots__ = ("_m", "_e")

    def __init__(self) -> None:
        self._m = 0.0              # mantissa
        self._e = 0                # binary exponent, plain int

    # -- accumulate --------------------------------------------------------- #
    def add(self, term: float, _frexp=math.frexp, _ldexp=math.ldexp,
            _isfinite=math.isfinite) -> None:
        """Add one term.  The default arguments bind the C functions as locals,
        which is ~60 ns cheaper per term than attribute lookups."""
        if term == 0.0:
            return                                     # adding zero is a no-op
        if not _isfinite(term):
            raise OverflowError("a single series term is not a finite float")

        mt, et = _frexp(term)                          # term = mt * 2**et
        m = self._m
        if m == 0.0:                                   # first non-zero term
            self._m, self._e = mt, et
            return

        e = self._e
        if et >= e:                                    # the new term is the bigger one
            m = mt + _ldexp(m, e - et)
            e = et
        else:                                          # the sum is the bigger one
            m += _ldexp(mt, et - e)

        # Renormalise only when |m| crossed 1.0.  Any mantissa in (0, 2) keeps
        # the same 53 bits, so the skipped case is bit-identical and saves one
        # frexp() per term.
        if m >= 1.0 or m <= -1.0:
            m, d = _frexp(m)
            e += d
        self._m, self._e = m, e

    # -- read out ----------------------------------------------------------- #
    def log_abs(self) -> float:
        """ln|S| without ever forming S -- works even beyond 1e308."""
        if self._m == 0.0:
            return -math.inf
        return math.log(abs(self._m)) + self._e * 0.6931471805599453

    def value(self) -> float:
        """Collapse to a float.  OverflowError only if |S| > DBL_MAX."""
        if self._m == 0.0:
            return 0.0
        return math.ldexp(self._m, self._e)

    def __float__(self) -> float:
        return self.value()

    def __repr__(self) -> str:
        return f"ScaledSum(m={self._m!r}, e={self._e})"
