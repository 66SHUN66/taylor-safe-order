#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assertions for every claim the README makes.

Runs with plain `python tests/test_accuracy.py` and is also pytest-compatible.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaled_sum import ScaledSum                                        # noqa: E402
from taylor_auto import (                                               # noqa: E402
    scan, evaluate_scaled, exp_terms, sin_terms, asymptotic_euler_terms,
)
from taylor_order import (                                              # noqa: E402
    OrderProblem, check_conservative_rho, check_interval_on_V, check_lipschitz,
)
from taylor_overflow import exp_reduced, sin_cos                        # noqa: E402

U = sys.float_info.epsilon / 2.0


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


# --------------------------------------------------------------------------- #
# 1.  Reduction-based evaluation
# --------------------------------------------------------------------------- #
def test_exp_reduced_matches_libm():
    for x in (0.0, 1.0, 10.0, 100.0, 300.0, 700.0, 709.0):
        v, n = exp_reduced(x)
        assert rel(v, math.exp(x)) <= 5e-16, (x, n, rel(v, math.exp(x)))
        assert n <= 20                                    # ~10-13 terms in practice


def test_exp_reduced_overflow_is_honest():
    try:
        exp_reduced(710.0)
    except OverflowError:
        return
    raise AssertionError("exp(710) must raise, not return inf")


def test_sin_cos_across_magnitudes():
    for x in (0.5, 2.0, 5.0, 100.0, 1e6):
        s, c, n_s, n_c = sin_cos(x)
        assert rel(s, math.sin(x)) <= 1e-15, (x, rel(s, math.sin(x)))
        assert rel(c, math.cos(x)) <= 1e-15, (x, rel(c, math.cos(x)))
        assert n_s <= 20 and n_c <= 20                    # reduction keeps it short


# --------------------------------------------------------------------------- #
# 2.  Order detection
# --------------------------------------------------------------------------- #
def test_scan_small_argument():
    r = scan(exp_terms(1.0))
    assert 10 <= r.n_star <= 20, r.n_star
    assert rel(r.value, math.e) <= 1e-15
    assert r.n_plateau == r.n_star + 1                    # optimum = last useful term
    assert r.verdict() == "OK"


def test_scan_large_argument():
    r = scan(exp_terms(700.0))
    assert 900 <= r.n_star <= 940, r.n_star
    v = evaluate_scaled(exp_terms(700.0), r.n_star)
    assert rel(v, math.exp(700.0)) <= 1e-13


def test_scan_reports_overflow_instead_of_inf():
    r = scan(exp_terms(710.0))
    assert r.n_overflow is not None
    assert r.n_plateau is None                            # never settled before the wall
    assert r.verdict().startswith("FAIL")
    assert not r.usable


def test_divergent_series_optimal_truncation():
    """The detector must land on the brute-force argmin for an asymptotic series."""
    x = 0.1
    ref = 0.9156333394                                    # Simpson, ~1e-9 accurate
    r = scan(asymptotic_euler_terms(x))
    assert r.n_star == 9, r.n_star                        # minimum-term rule
    err_at_star = rel(evaluate_scaled(asymptotic_euler_terms(x), r.n_star), ref)
    assert err_at_star <= 5e-4, err_at_star
    # one term further must be no better
    err_next = rel(evaluate_scaled(asymptotic_euler_terms(x), r.n_star + 1), ref)
    assert err_next >= err_at_star * 0.95, (err_at_star, err_next)


def test_sin_taylor_is_useless_at_large_x():
    r = scan(sin_terms(100.0), max_terms=200_000)
    assert r.est_rel >= 1.0                               # no order helps
    assert r.verdict().startswith("FAIL")
    assert r.peak_term > 1e40                             # the cause: a 1e42 peak


# --------------------------------------------------------------------------- #
# 3.  The accumulator
# --------------------------------------------------------------------------- #
def test_scaled_sum_matches_exact_sum():
    random.seed(7)
    worst = 0.0
    for _ in range(50):
        terms = [random.choice([1, -1]) * 10.0 ** random.uniform(-200, 100)
                 for _ in range(60)]
        acc = ScaledSum()
        for t in terms:
            acc.add(t)
        ref = math.fsum(terms)
        if ref:
            worst = max(worst, rel(acc.value(), ref))
    assert worst <= 4 * sys.float_info.epsilon, worst      # <= ~4 ulp of fsum


def test_scaled_sum_survives_beyond_double_range():
    """Terms whose total is at / past DBL_MAX: accumulate, then overflow once."""
    acc = ScaledSum()
    for _ in range(10):
        acc.add(1e307)
    # 10 * 1e307 = 1e308, which is still representable (1 ulp of rounding)
    assert rel(acc.value(), 1e308) <= 1e-15, acc.value()

    acc = ScaledSum()
    for _ in range(18):                                   # 1.8e308 > DBL_MAX
        acc.add(1e307)
    try:
        acc.value()
    except OverflowError:
        return
    raise AssertionError("18 * 1e307 must overflow at read-out, not earlier")


# --------------------------------------------------------------------------- #
# 4.  The theorem side
# --------------------------------------------------------------------------- #
def _geometric_case() -> OrderProblem:
    """f = 1/(1-z), rho = 0.9, V = 10, r = 0.5, eps = 1e-6."""
    return OrderProblem(r=0.5, rho=0.9, V=10.0, eps=1e-6)


def test_n_min_is_conservative_and_valid():
    p = _geometric_case()
    n_min = p.n_min()
    assert n_min == 28, n_min                              # regression against the paper
    # the certificate really bounds the true remainder
    true_r = 0.5 ** (n_min + 1) / (1.0 - 0.5)
    assert true_r <= p.error_bound(n_min) <= p.eps
    # brute force the truly needed order and check it is not larger
    n_actual = next(n for n in range(200) if 0.5 ** (n + 1) / 0.5 <= p.eps)
    assert n_actual <= n_min


def test_feasible_region_and_overflow_bound():
    p = _geometric_case()
    assert p.feasible() and p.order_star() == p.n_min()
    tight = OrderProblem(r=0.5, rho=0.9, V=10.0, eps=1e-6, M=15.0)
    cert = tight.certificate()
    assert cert["B_inf"] > 15.0                            # the bound says infeasible
    assert not tight.feasible()
    assert 1.0 / (1.0 - 0.5) < 15.0                        # ... but the truth fits


def test_lemmas_and_lipschitz():
    p = _geometric_case()
    assert check_conservative_rho(p, 0.85)["holds"]
    assert check_interval_on_V(p, 8.0, 12.0)["holds"]
    res = check_lipschitz(p, 0.85, 0.95)
    assert res["holds"], res
    assert 60.0 < res["C"] < 66.0                          # ~63.09 measured


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{len([n for n in globals() if n.startswith('test_')])} tests, "
          f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
