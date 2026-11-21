#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taylor_overflow.py
==================

A dependency-free toolkit that answers the two questions that break naive
Taylor expansion in double precision:

    (1) HOW MANY TERMS (the order N) does the expansion need?
        ->  exp_order()            : analytic count, computed in LOG space
    (2) HOW DO I EVALUATE IT WITHOUT OVERFLOWING?
        ->  ScaledSum              : partial sum stored as  m * 2**e
        ->  exp_reduced()          : x = k*ln2 + r  argument reduction
        ->  sin_cos()              : x = k*(pi/2) + r quadrant reduction

Why a naive textbook loop fails
-------------------------------
    S = sum_{k=0..N} x**k / k!

breaks in three *different* ways, and each one needs its own fix:

    a) the term itself overflows      : x**k / k! is computed via x**k first,
                                        which leaves the float range long before
                                        the quotient does (e.g. 200**200 = 1e460)
    b) the running sum overflows      : terms pass through a huge peak even when
                                        the final answer is small
    c) cancellation                   : for e**(-x) the terms reach 1e306 while
                                        the answer is 1e-300 -> ~600 digits lost

Fixes used here:
    (a) -> never materialise x**k / k!; update the term recursively,
           t_k = t_(k-1) * x / k  (divide first, so it never overshoots)
    (b) -> accumulate in a scaled (mantissa, exponent) form, ScaledSum
    (c) -> do not sum e**(-x) at all; reduce and take the reciprocal instead

Textbook warning for the order count
------------------------------------
    For e**x the Lagrange remainder gives
        |R_N(x)| <= e**x * x**(N+1) / (N+1)!  =  e**x * |t_(N+1)|
    so the stopping test "next term <= eps * e**x" is a *proof*, not a guess.

Author: 66SHUN66  (companion to taylor_auto.py / taylor_order.py)
"""

from __future__ import annotations

import math
import sys
from typing import Tuple
from scaled_sum import ScaledSum

# --------------------------------------------------------------------------- #
# 0.  Constants that describe the limits of IEEE-754 double
# --------------------------------------------------------------------------- #
DBL_MAX_LOG = math.log(sys.float_info.max)      # ≈  709.7827 : e**x overflows above
DBL_MIN_LOG = math.log(sys.float_info.min)      # ≈ -708.3964 : e**x underflows below
EPS = sys.float_info.epsilon                    # ≈  2.22e-16 : 1 ulp at 1.0

# ln 2 split into two parts so that  k*ln2  can be subtracted without
# losing the low bits of x  (this is the classic "double-double" trick)
LN2 = 0.6931471805599453
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
LOG2E = 1.4426950408889634

# pi/2 split the same way, for the sin/cos quadrant reduction
PIO2 = 1.5707963267948966
PIO2_1 = 1.57079632673412561417e+00
PIO2_1T = 6.07710050650619224932e-11


# --------------------------------------------------------------------------- #
# 1.  The scaled accumulator lives in scaled_sum.py (imported above)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 2.  How many terms?  — the order count, done in log space
# --------------------------------------------------------------------------- #
def log_abs_term_exp(x: float, n: int) -> float:
    """ln |t_n|  for  t_n = x**n / n!  , computed with lgamma (never overflows).

    ln|t_n| = n*ln x - ln(n!)  and  ln(n!) = lgamma(n+1).
    For n >= x this function is strictly decreasing  (ratio = x/(n+1) < 1),
    which is what makes the binary search below legal.
    """
    return n * math.log(x) - math.lgamma(n + 1.0)


def peak_log_term_exp(x: float) -> float:
    """ln of the largest term of the e**x series ~= e**x / sqrt(2*pi*x).

    The terms grow while n < x and decay afterwards, so the peak sits at n ≈ x.
    If this value exceeds DBL_MAX_LOG the *term* cannot be held as a float,
    which is the precondition for needing ScaledSum at all.
    """
    n = max(int(math.floor(x)), 0)
    return log_abs_term_exp(x, n)


def exp_order(x: float, rel_eps: float = EPS) -> int:
    """Smallest N such that  P_N(x) = sum_{k=0..N} x**k/k!  has

        |R_N(x)| <= rel_eps * e**x      (relative error guarantee)

    Logic:  R_N is dominated by t_{N+1}, and t_{N+1} <= t_N * x/(N+1), so it
    is enough to require  |t_N| <= rel_eps * e**x  with N past the peak.

    Everything is decided in log space:
        n*ln x - ln(n!) <= x + ln(rel_eps)
    so no term is ever materialised and the count itself cannot overflow.
    """
    if x == 0.0:
        return 0                                       # e**0 = 1 exactly
    if x < 0.0:                                        # e**(-x): same term sizes
        return exp_order(-x, rel_eps)

    log_target = x + math.log(rel_eps)                 # ln(eps * e**x)

    lo = max(1, int(math.ceil(x)))                     # from here on: decreasing
    hi = lo
    while log_abs_term_exp(x, hi) > log_target:        # expand until the test passes
        hi = 2 * hi + 2
        if hi > 1_000_000:
            raise ArithmeticError("order search diverged (x too large?)")

    while lo < hi:                                     # classic bisection
        mid = (lo + hi) // 2
        if log_abs_term_exp(x, mid) <= log_target:
            hi = mid                                   # mid already good enough
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------- #
# 3.  Two ways to evaluate e**x
# --------------------------------------------------------------------------- #
def exp_direct(x: float, rel_eps: float = EPS) -> Tuple[float, int]:
    """e**x straight from its Taylor series, order chosen automatically.

    Returns (value, order).  Overflow-safe for |x| < 710, but the accuracy
    degrades slowly because ~x terms are added (each one rounds).
    """
    order = exp_order(x, rel_eps)

    s = ScaledSum()
    t = 1.0                                            # t_0 = 1
    s.add(t)
    for n in range(1, order + 1):
        t = t / n * x                                  # divide FIRST: t/n*x never
        s.add(t)                                        # overshoots the true term
    return s.value(), order


def exp_reduced(x: float, rel_eps: float = EPS) -> Tuple[float, int]:
    """e**x via argument reduction:  x = k*ln2 + r  ->  e**x = 2**k * e**r.

    |r| <= ln2/2 ≈ 0.347, so the series needs ~16 terms instead of ~900 and
    there is no cancellation.  This is what libm actually does.

    Returns (value, order).  Raises OverflowError when the true answer does
    not fit in a double (|x| > 709.78) — which is the correct behaviour.
    """
    k = int(math.floor(x * LOG2E + 0.5))               # k = round(x / ln2)
    r = (x - k * LN2_HI) - k * LN2_LO                  # two-step: keeps low bits

    s = ScaledSum()
    t = 1.0                                            # t_0 = 1
    s.add(t)
    order = 0
    for n in range(1, 21):                             # 20 terms is plenty here
        t = t * r / n
        s.add(t)
        order = n
        if abs(t) <= rel_eps:                          # |R_n| <= |t_(n+1)| ~ |t_n|
            break

    try:
        return math.ldexp(s.value(), k), order          # 2**k, via the exponent
    except OverflowError:
        raise OverflowError(
            f"exp({x}) ~ 1e{ x / math.log(10) :.0f} exceeds DBL_MAX (1.8e308)"
        ) from None


# --------------------------------------------------------------------------- #
# 4.  sin / cos with quadrant reduction
# --------------------------------------------------------------------------- #
def _sin_series(r: float, eps: float) -> Tuple[float, int]:
    """sin(r) for |r| <= pi/4.  Alternating series -> |R| <= |next term|."""
    t = r                                              # t_1
    s = r                                              # partial sum
    n = 1                                              # highest power so far
    while True:
        t_next = -t * r * r / ((n + 1) * (n + 2))      # t_(n+2)
        if abs(t_next) <= eps:                         # next term bounds the tail
            return s, n
        t = t_next
        s += t
        n += 2


def _cos_series(r: float, eps: float) -> Tuple[float, int]:
    """cos(r) for |r| <= pi/4.  Alternating series, same tail bound."""
    t = 1.0                                            # t_0
    s = 1.0
    n = 0
    while True:
        t_next = -t * r * r / ((n + 1) * (n + 2))      # t_(n+2)
        if abs(t_next) <= eps:
            return s, n
        t = t_next
        s += t
        n += 2


def sin_cos(x: float, eps: float = EPS) -> Tuple[float, float, int, int]:
    """(sin x, cos x, order_of_sin, order_of_cos) with  x = k*(pi/2) + r.

    Why reduce?  sin/cos of a large argument is a *different* disaster: x itself
    is fine, but x mod 2*pi loses most of its bits (catastrophic cancellation),
    and Taylor on a large x needs a huge order.  Reducing to |r| <= pi/4 fixes
    both at once — the series then converges in ~8 terms.

    Caveat: the reduction here uses a 2-word split of pi/2, which is exact while
    k*PIO2_1 stays exactly representable, i.e. |x| <~ 1e7 (measured: 0 ulp error
    up to 1e7, ~1e-9 relative error at 1e8, ~1e-7 at 1e10).  Beyond that you need
    Payne-Hanek multi-word reduction, which is what math.sin() uses.
    """
    k = int(math.floor(x / PIO2 + 0.5))                # quadrant: x ≈ k*(pi/2)
    r = (x - k * PIO2_1) - k * PIO2_1T                 # remainder, |r| <= pi/4

    s_val, s_ord = _sin_series(r, eps)                 # sin(r), needs k % 4 == 0
    c_val, c_ord = _cos_series(r, eps)                 # cos(r)

    q = k % 4                                          # which quadrant we are in
    if q == 0:
        return s_val, c_val, s_ord, c_ord
    if q == 1:
        return c_val, -s_val, c_ord, s_ord             # sin(x) = cos(r)
    if q == 2:
        return -s_val, -c_val, s_ord, c_ord
    return -c_val, s_val, c_ord, s_ord


def sin(x: float, eps: float = EPS) -> float:
    return sin_cos(x, eps)[0]


def cos(x: float, eps: float = EPS) -> float:
    return sin_cos(x, eps)[1]


# --------------------------------------------------------------------------- #
# 5.  The "textbook" version, kept around on purpose, to show what breaks
# --------------------------------------------------------------------------- #
def exp_naive_factorial(x: float, order: int) -> float:
    """sum x**k / k! written the way the formula looks.  Do not use this."""
    total = 0.0
    for k in range(order + 1):
        total += x ** k / math.factorial(k)            # x**k is built first -> boom
    return total


# --------------------------------------------------------------------------- #
# 6.  How to adapt this to any other expansion (recipe, not code)
# --------------------------------------------------------------------------- #
#  Suppose your function is  f(x) = sum t_n(x)  with  t_n = c_n * x**n.
#
#  Step 1  write the term in LOG form, never with factorials:
#              log_abs_term(n) = log|c_n| + n*log(x)        (c_n = 1/n! -> -lgamma(n+1))
#          log|c_n| itself must also come from log space (lgamma, log of a product).
#
#  Step 2  find the peak of |t_n|:  it sits where  |t_(n+1)/t_n| = 1.
#          Because of the peak, "first term below eps" is NOT the stop rule —
#          you must pass the peak first, then stop.
#
#  Step 3  count the order in log space (binary search, as in exp_order):
#              smallest N with  log|t_N| <= log(eps * |S|)
#          |S| is estimated in closed form (e.g. |S| = e**x) or via ScaledSum.log_abs().
#
#  Step 4  evaluate:  update the term recursively (divide before multiply), and
#          accumulate with ScaledSum so a huge peak cannot overflow.
#
#  Step 5  if the interval is wide (|x| large), reduce the argument first
#          (x = k*ln2 + r for exp, x = k*pi/2 + r for trig), otherwise the order
#          grows like |x| and cancellation eats your digits.
#
#  Step 6  when the series alternates AND its terms are much larger than the
#          answer (e^-x, ln(1+x) near x=-1), no scaling can save you: use an
#          identity to turn it into a benign computation (1/e**x, log1p, ...).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 7.  Demo / self-test
# --------------------------------------------------------------------------- #
def _rel_err(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


def _row(cols, widths) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def main() -> None:
    print("=" * 96)
    print("How many terms does e**x need, and does the evaluation survive?")
    print("=" * 96)
    w = (10, 14, 14, 14, 16)
    print(_row(["x", "peak ln|t_n|", "order N", "terms", "peak overflows?"], w))
    print("-" * 96)
    for x in (1.0, 10.0, 50.0, 200.0, 500.0, 700.0, 709.0, 710.0):
        order = exp_order(x)
        peak = peak_log_term_exp(x)
        flag = "YES" if peak > DBL_MAX_LOG else "no"
        print(_row([f"{x:g}", f"{peak:.2f}", f"{order}", f"{order + 1}", flag], w))

    print()
    print("=" * 96)
    print("Accuracy: naive factorial loop vs direct Taylor vs reduced argument")
    print("=" * 96)
    w2 = (8, 26, 20, 20, 12)
    print(_row(["x", "naive x**k/k!", "direct Taylor", "range-reduced", "math.exp"], w2))
    print("-" * 96)
    for x in (1.0, 10.0, 50.0, 200.0, 500.0, 700.0):
        ref = math.exp(x)

        try:
            order = exp_order(x)
            naive = f"rel err {_rel_err(exp_naive_factorial(x, order), ref):.2e}"
        except OverflowError:
            naive = "OverflowError"

        try:
            v1, n1 = exp_direct(x)
            direct = f"n={n1}, err {_rel_err(v1, ref):.2e}"
        except OverflowError:
            direct = "OverflowError"

        try:
            v2, n2 = exp_reduced(x)
            reduced = f"n={n2}, err {_rel_err(v2, ref):.2e}"
        except OverflowError:
            reduced = "OverflowError"

        print(_row([f"{x:g}", naive, direct, reduced, f"{ref:.3e}"], w2))

    print()
    print("=" * 96)
    print("sin / cos at large arguments: reduction keeps the order small")
    print("=" * 96)
    w3 = (10, 18, 18, 12, 12)
    print(_row(["x", "sin (reduced)", "cos (reduced)", "order sin", "order cos"], w3))
    print("-" * 96)
    for x in (0.5, 2.5, 7.0, 100.0, 1000.0, 1e8):
        s, c, os_, oc_ = sin_cos(x)
        print(_row(
            [f"{x:g}",
             f"{s:.15f}  ({_rel_err(s, math.sin(x)):.1e})",
             f"{c:.15f}  ({_rel_err(c, math.cos(x)):.1e})",
             f"{os_}", f"{oc_}"], w3))

    print()
    print("=" * 96)
    print("Edge cases: honest failure instead of silent garbage")
    print("=" * 96)
    for x in (709.0, 710.0, -745.0, -750.0):
        try:
            v, n = exp_reduced(x)
            print(f"  exp_reduced({x:g}) = {v:.6e}   (n={n})")
        except OverflowError as e:
            print(f"  exp_reduced({x:g}) -> OverflowError: {e}")

    # e**(-x) is NOT solvable by scaling alone: the terms reach 1e306 while
    # the answer is 1e-306, i.e. ~600 decimal digits of cancellation.  The
    # only sane route is the reciprocal.
    x = 300.0
    print(f"\n  e**-{x:g}: direct series would need "
          f"~{peak_log_term_exp(x) / math.log(10):.0f} decimal digits of cancellation")
    print(f"           1/exp_reduced({x:g}) = {1.0 / exp_reduced(x)[0]:.6e}  "
          f"(err {_rel_err(1.0 / exp_reduced(x)[0], math.exp(-x)):.2e})")


if __name__ == "__main__":
    main()
