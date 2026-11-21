#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taylor_auto.py
==============

Automatic ORDER DETECTION for a Taylor series: stop at the order where the
result is MOST accurate while the evaluation still fits in double precision.

This is a different question from "smallest order meeting a tolerance"
(that is Theorem 1 in taylor_order.py).  Here the issue is that accuracy does
NOT improve forever:

    total error  ~  |tail|            (truncation: shrinks as N grows)
                 +  (N+1)*u*sum|t_k|  (rounding: GROWS as N grows)

so there is an interior optimum N*.  For divergent (asymptotic) series this
optimum is the classic "minimum term" rule; for convergent series it is the
point where the term drops below half an ulp of the running sum.

Three stopping mechanisms, all implemented in scan():

    1. rounding floor   : |t_k| <= u*|S_k|  -> the term can no longer change
                          the double result (n_plateau)
    2. growth detector  : |t_{k+1}| >= |t_k|  -> the tail is not summable,
                          the minimum term is already behind us
    3. overflow guard   : below the last order k with both |t_k| and |S_k|
                          finite (n_overflow); beyond it everything is inf

What "most accurate" means here is stated explicitly: the argmin of
    est(N) = |t_{N+1}| / (1 - r)  +  (N+1)*u*B_N ,      r = |t_{N+1}/t_N|
which is a bound, not a guess, for alternating series (Leibniz) and a
geometric-tail estimate otherwise.

Usage
-----
    from taylor_auto import scan, evaluate_scaled, exp_terms

    r = scan(exp_terms(10.0))                  # find the best order
    value = evaluate_scaled(exp_terms(10.0), r.n_star)   # evaluate it safely
    print(r.n_star, r.est_rel, r.verdict())

To plug in your own expansion, write ONE generator that yields t_0, t_1, ...
recursively -- never `a_k * x**k`, because that product is what overflows:

    def my_terms(x):
        t, k = <t_0>, 0
        while True:
            yield t
            k += 1
            t = <t_k expressed through t_(k-1)>
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from scaled_sum import ScaledSum

DBL_MAX = sys.float_info.max          # 1.797e308
U = sys.float_info.epsilon / 2.0      # u = 2**-53 = 1.11e-16 (half ulp at 1.0)


# --------------------------------------------------------------------------- #
# 0.  The scaled accumulator lives in scaled_sum.py (imported above)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 1.  The scan result
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    n_star: int                    # order with minimal estimated total error
    value: float                   # partial sum at n_star (plain double sum)
    est_rel: float                 # estimated RELATIVE error at n_star
    est_abs: float                 # estimated ABSOLUTE error at n_star
    n_overflow: int | None         # first order where a term/sum left the range
    n_plateau: int | None          # first order whose term is below 0.5 ulp(S)
    peak_term: float               # largest |t_k| seen
    last_k: int                    # last order actually visited

    @property
    def usable(self) -> bool:
        """An answer worth returning (WEAK is still a number; FAIL is not)."""
        return not self.verdict().startswith("FAIL")

    def verdict(self) -> str:
        """Three mutually exclusive diagnoses, in order of severity."""
        if self.n_overflow is not None and (self.n_plateau is None
                                            or self.n_star < self.n_plateau):
            return (f"FAIL: range ends at term {self.n_overflow}, plateau "
                    f"{self.n_plateau if self.n_plateau is not None else 'never reached'}")
        if self.est_rel >= 1.0:
            return "FAIL: no order helps (est. rel. error >= 1, reduce the argument)"
        if self.est_rel > 1e-10:
            return "WEAK: rounding, not truncation, sets the accuracy"
        return "OK"


# --------------------------------------------------------------------------- #
# 2.  The scanner
# --------------------------------------------------------------------------- #
def scan(terms, u: float = U, max_terms: int = 1_000_000,
         patience: int = 100, plateau_guard: int = 16) -> Result:
    """Walk t_0, t_1, ... and return the order with the smallest error estimate.

    `terms` is any iterator of floats; the caller is responsible for generating
    them RECURSIVELY (t_k from t_{k-1}), never as a_k and r**k separately --
    otherwise the generator itself overflows before the series has a chance to.

    patience: stop after this many steps without a new best, once the
    estimate has grown by more than 4x.  Purely a speed guard; the answer is
    already fixed at that point.

    plateau_guard: once |t_k| <= u*|S_k| the term is below half an ulp of the
    running sum, so no further addition can change the double result.  After
    plateau_guard more terms we stop: past that point the estimate can only
    grow.  This is what keeps e**700 down to ~930 terms instead of ~2500.
    """
    isfinite = math.isfinite
    abs_ = abs
    inf = math.inf
    gen = iter(terms)
    nxt = gen.__next__                       # bound method: one lookup per term
    try:
        t = nxt()
    except StopIteration:
        raise ValueError("empty term sequence") from None

    S = 0.0                      # running partial sum, plain double
    B = 0.0                      # sum |t_k|, saturating at DBL_MAX
    best_est = inf               # smallest relative error estimate seen
    k_star = 0                   # order that achieved it
    best_val = 0.0               # partial sum at that order
    best_abs = inf               # absolute estimate at that order
    n_ovf: int | None = None     # first order that left the float range
    n_flat: int | None = None    # first order whose term is below 0.5 ulp(S)
    peak = 0.0
    k_best = -1
    k = -1

    while True:
        k += 1
        if k > max_terms:
            raise RuntimeError(f"scan exceeded max_terms = {max_terms}")

        if not isfinite(t):                       # cannot even hold the term
            n_ovf = k
            break

        at = abs_(t)
        if at > peak:
            peak = at

        # lookahead: needed for the truncation estimate
        try:
            t_next = nxt()
            nxt_bad = not isfinite(t_next)        # arithmetic overflowed to inf
        except StopIteration:
            t_next, nxt_bad = 0.0, False
        except OverflowError:                     # math.exp & friends raise
            t_next, nxt_bad = inf, True

        newS = S + t
        if not isfinite(newS):                    # the SUM leaves the range
            n_ovf = k
            break
        S = newS
        B += at
        if not isfinite(B):
            B = DBL_MAX                           # saturate: rounding is hopeless

        # ---- error model ------------------------------------------------- #
        an = abs_(t_next)
        if at == 0.0:
            trunc = an                            # zero coefficient: skip it
        elif an == 0.0:
            trunc = at                            # a *zero next* term proves nothing
        elif an >= at:                            # still growing: no tail bound
            trunc = an
        else:                                     # geometric tail
            trunc = an / (1.0 - an / at)
        est_abs = trunc + (k + 1) * u * B
        sabs = abs_(S)
        est_rel = est_abs / sabs if sabs else est_abs

        if est_rel < best_est:                    # scalars instead of a tuple
            best_est, k_star, best_val, best_abs, k_best = est_rel, k, S, est_abs, k

        if n_flat is None and k > 0 and 0.0 < at <= u * sabs:
            # A *structurally zero* coefficient (sin's even terms) is not a
            # plateau -- hence the 0.0 < at guard.
            n_flat = k

        # ---- stop conditions --------------------------------------------- #
        if nxt_bad:                               # the very next term is inf
            n_ovf = k + 1
            break
        if n_flat is not None and k >= n_flat + plateau_guard:
            break                                 # sum already at its noise floor
        if k - k_best > patience and est_rel > 4.0 * best_est:
            break                                 # optimum is safely behind us
        if t_next == 0.0 and at == 0.0:
            break                                 # series ended exactly
        t = t_next

    return Result(
        n_star=k_star, value=best_val, est_rel=best_est, est_abs=best_abs,
        n_overflow=n_ovf, n_plateau=n_flat, peak_term=peak, last_k=k,
    )


def evaluate_plain(terms, n: int) -> float:
    """Replay the generator and sum the first n+1 terms in plain double."""
    nxt = iter(terms).__next__
    s = 0.0
    try:
        for _ in range(n + 1):
            s += nxt()
    except StopIteration:
        raise ValueError(f"term sequence ended before order {n}") from None
    return s


def evaluate_scaled(terms, n: int) -> float:
    """Same partial sum, but through ScaledSum: no intermediate can overflow."""
    nxt = iter(terms).__next__
    acc = ScaledSum()
    add = acc.add
    try:
        for _ in range(n + 1):
            add(nxt())
    except StopIteration:
        raise ValueError(f"term sequence ended before order {n}") from None
    return acc.value()


# --------------------------------------------------------------------------- #
# 3.  Term generators (recursive, never materialising r**k or k!)
# --------------------------------------------------------------------------- #
def exp_terms(x: float):
    """e**x = sum x**k/k!,  t_k = t_(k-1) * x / k  (divide first)."""
    t = 1.0
    k = 0
    while True:
        yield t
        k += 1
        t = t / k * x


def sin_terms(x: float):
    """sin x = sum (-1)^j x^(2j+1)/(2j+1)!, two-step recurrence."""
    tm2, tm1 = 0.0, x                    # t_0, t_1
    yield tm2
    yield tm1
    k = 2
    while True:
        t = -tm2 * x * x / ((k - 1) * k)  # t_k from t_(k-2)
        yield t
        tm2, tm1 = tm1, t
        k += 1


def asymptotic_euler_terms(x: float):
    """I(x) = int_0^inf e^-t/(1+xt) dt ~ sum (-1)^k k! x^k  (divergent!)."""
    t = 1.0
    k = 0
    while True:
        yield t
        k += 1
        t = -t * x * k                   # t_k = -t_(k-1) * x * k, no factorial


# --------------------------------------------------------------------------- #
# 4.  Demo
# --------------------------------------------------------------------------- #
def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def _row(cols, widths) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def i_reference(x: float, length: float = 60.0, steps: int = 12000) -> float:
    """Composite Simpson for I(x) = int_0^L e^-t/(1+xt) dt  (~1e-12 accurate)."""
    h = length / steps
    total = 0.0
    for i in range(steps + 1):
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += w * math.exp(-i * h) / (1.0 + x * i * h)
    return total * h / 3.0


def main() -> None:
    print("=" * 100)
    print("Block 1  e**x : the detected order tracks x, and stops at the overflow cliff")
    print("=" * 100)
    w = (7, 7, 8, 12, 14, 11, 44)
    print(_row(["x", "N*", "plateau", "est rel err", "rel err vs libm", "n_overflow", "verdict"], w))
    print("-" * 100)
    for x in (1.0, 5.0, 10.0, 100.0, 300.0, 700.0, 709.0, 710.0, 800.0):
        r = scan(exp_terms(x))
        est = "nan" if math.isnan(r.est_rel) else f"{r.est_rel:.2e}"
        try:
            err = f"{_rel_err(r.value, math.exp(x)):.2e}"
        except OverflowError:
            err = "n/a (libm overflows)"
        plateau = "none" if r.n_plateau is None else r.n_plateau
        ovf = "none" if r.n_overflow is None else r.n_overflow
        print(_row([f"{x:g}", r.n_star, plateau, est, err, ovf, r.verdict()], w))
    print("  note: at x=710 the answer 2.3e308 does not fit in a double at all,")
    print("        and the term range ends before the tail settles -> FAIL, not a number.")
    print("  note: at x=800 the sum itself leaves the range at term 458, long before")
    print("        convergence -> no order exists, whatever N you pick.")
    print("  note: WEAK/OK rows keep N* <= plateau, so evaluate_scaled() can be used")
    print("        to squeeze the last ulps of head-room without overflowing.")

    print()
    print("=" * 100)
    print("Block 2  plain Taylor sin x : detection reports where arithmetic itself fails")
    print("=" * 100)
    w2 = (8, 8, 12, 16, 34)
    print(_row(["x", "N*", "est rel err", "rel err vs libm", "why"], w2))
    print("-" * 100)
    for x in (0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        r = scan(sin_terms(x), max_terms=200_000)
        why = ("no order helps: terms reach %.1e, rounding eats the answer"
               % r.peak_term) if r.est_rel >= 1.0 else "series itself is fine"
        print(_row([f"{x:g}", r.n_star, f"{r.est_rel:.2e}",
                    f"{_rel_err(r.value, math.sin(x)):.2e}", why], w2))
    print("  -> x=100: peak term ~1e42, so the partial sum carries ~1e26 of rounding")
    print("     error; NO order can fix that.  Range reduction reduces x to |r|<=pi/4")
    print("     and gets 1e-16 with 13 terms (see sin_cos() in taylor_overflow.py).")

    print()
    print("=" * 100)
    print("Block 3  divergent series: the detector finds the classic optimal truncation")
    print("=" * 100)
    x = 0.1
    ref = i_reference(x)
    print(f"  I(x) = integral_0^inf e^-t/(1+xt) dt  ~  sum (-1)^k k! x^k,   x = {x}")
    print(f"  reference value (Simpson) = {ref:.12f}")
    w3 = (5, 16, 16, 16, 10)
    print(_row(["k", "t_k", "S_k (double)", "true rel err", ""], w3))
    print("-" * 100)
    r = scan(asymptotic_euler_terms(x))
    gen = asymptotic_euler_terms(x)
    s = 0.0
    for k in range(0, min(r.n_star + 7, 21)):
        t = next(gen)
        s += t
        mark = "<-- N*" if k == r.n_star else ""
        print(_row([k, f"{t:.6e}", f"{s:.6e}", f"{_rel_err(s, ref):.2e}", mark], w3))
    print(f"  detector N* = {r.n_star}, model est = {r.est_rel:.2e}, "
          f"achieved = {_rel_err(evaluate_plain(asymptotic_euler_terms(x), r.n_star), ref):.2e}")
    print("  error falls until the smallest term, then rises: adding terms makes it WORSE.")
    print("  the detector stops exactly at that minimum -- this is why 'more terms'")
    print("  is not 'more accurate' for asymptotic expansions.")

    print()
    print("=" * 100)
    print("Block 4  the evaluation guarantee")
    print("=" * 100)
    for x in (700.0, 709.0):
        r = scan(exp_terms(x))
        plain = evaluate_plain(exp_terms(x), r.n_star)
        scaled = evaluate_scaled(exp_terms(x), r.n_star)
        print(f"  e^{x:g}: N*={r.n_star}  plain={plain:.6e}  scaled={scaled:.6e}  "
              f"libm={math.exp(x):.6e}  rel err(scaled)={_rel_err(scaled, math.exp(x)):.2e}")


if __name__ == "__main__":
    main()
