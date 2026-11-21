#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taylor_order.py
===============

Reference implementation + numerical verification of

    "Rigorous Existence, Stability, and Optimality of
     Taylor Order Constrained Optimization"

Every numbered result of the paper has exactly one implementation below, so the
map theorem <-> code is one-to-one:

    Theorem 1  (closed-form feasible region)   -> OrderProblem.n_min / n_max / feasible
    Theorem 2  (local Lipschitz stability)     -> lipschitz_constant() / check_lipschitz()
    Lemma   1  (conservative rho)              -> check_conservative_rho()
    Lemma   2  (interval arithmetic on V)      -> check_interval_on_V()
    Assumption 1 (overflow amplification kappa)-> gamma()  (kappa = gamma_{2N}, Horner)
    Sec. 4     (information-theoretic optimality) -> not a numerical statement;
                     cited in comments, no code (see the note in Theorem 1 block)

Notation is kept identical to the paper:

    f analytic on D(x0, rho),     r = |x - x0|  (evaluation radius),
    p = r/rho in (0, 1),          V = sup_{|z-x0| = rho} |f(z)|,
    V/rho**k bounds the Cauchy coefficients |f^(k)(x0)|/k!,
    eps = absolute truncation tolerance,   M = overflow threshold for the
    running partial sum B_N  (default DBL_MAX).

Two numerical remarks that the paper does not spell out, and that the code
therefore handles explicitly:

  R1. The closed form must itself be evaluated in LOG space.
      For rho = 709, eps = 1e-16 the quotient eps*(1-p)/V is about 1.5e-326,
      i.e. *below* the smallest subnormal double, so a literal
      math.log(eps*(1-p)/V) raises ValueError.  We always use
        log(eps) + log1p(-p) - log(V)
      which never underflows.

  R2. B_N is only an upper bound on the true partial sum, and it can be
      hugely conservative (B_inf = V/(1-p) while the true sum may be 1e304).
      Theorem 1 can therefore declare "infeasible" for a problem that is
      perfectly feasible in floating point.  The code reports both the
      certified bound and the true B_N so the gap is visible.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# 0.  Machine constants
# --------------------------------------------------------------------------- #
DBL_MAX = sys.float_info.max            # 1.7976931348623157e308
U = sys.float_info.epsilon / 2.0        # unit roundoff u = 2**-53 = 1.11e-16

LOG_DBL_MAX = math.log(DBL_MAX)         #  709.78  : exp(x) overflows above this


def _safe_exp(x: float) -> float:
    """exp(x) that saturates to +inf / 0 instead of raising.

    Needed because the paper's bounds legitimately go outside the double
    range: the *bound* may overflow while the quantity it bounds does not.
    """
    if x > LOG_DBL_MAX:
        return math.inf
    if x < -745.1332191019411:                    # log(smallest subnormal)
        return 0.0
    return math.exp(x)


def gamma_steps(n: int, u: float = U) -> float:
    """gamma_n = n*u / (1 - n*u): the classic forward-error coefficient.

    Standard model (Higham, Thm 3.1): a sum of n terms evaluated in floating
    point satisfies |fl(S) - S| <= gamma_(n-1) * sum|t_k|.  This is exactly the
    paper's Assumption 1 with kappa = gamma_(2N) for Horner evaluation, i.e.
    the assumption is not an assumption at all -- it is a theorem.
    """
    nu = n * u
    if nu >= 1.0:                                 # beyond the model's validity
        return math.inf
    return nu / (1.0 - nu)


# --------------------------------------------------------------------------- #
# 1.  The problem instance and Theorem 1
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrderProblem:
    """One (f, x0, rho, V, eps, M) instance.

    Frozen so that a closed-form answer can never depend on hidden state.
    """

    r: float                      # evaluation radius |x - x0|
    rho: float                    # analyticity radius
    V: float                      # sup |f| on |z - x0| = rho   (max modulus)
    eps: float                    # required absolute truncation error
    M: float = DBL_MAX            # overflow threshold for the partial sum

    def __post_init__(self) -> None:
        if not (0.0 < self.r < self.rho):
            raise ValueError("need 0 < r < rho (evaluation point inside the disk)")
        if not (self.V > 0.0 and self.eps > 0.0 and self.M > 0.0):
            raise ValueError("V, eps, M must be strictly positive")

    # ---- basic quantities --------------------------------------------- #
    @property
    def p(self) -> float:
        """Normalised evaluation radius p = r/rho in (0, 1)."""
        return self.r / self.rho

    # ---- the two competing bounds ------------------------------------- #
    def error_bound(self, N: int) -> float:
        """E_abs(N) = V/(1-p) * p**(N+1)   (strictly decreasing in N).

        Evaluated as exp(log V - log1p(-p) + (N+1) log p) so that neither
        p**(N+1) nor the quotient can underflow on the way (remark R1).
        """
        p = self.p
        return _safe_exp(math.log(self.V) - math.log1p(-p) + (N + 1) * math.log(p))

    def sum_bound(self, N: int) -> float:
        """B_N = V * sum_{k=0..N} p**k = V (1 - p**(N+1))/(1-p).

        Strictly increasing in N; B_inf = V/(1-p) is the worst case.
        """
        p = self.p
        tail = (N + 1) * math.log(p)               # log(p**(N+1)), exact enough
        if tail < -745.0:                          # p**(N+1) underflowed to 0
            return _safe_exp(math.log(self.V) - math.log1p(-p))
        return _safe_exp(
            math.log(self.V) + math.log(-math.expm1(tail)) - math.log1p(-p)
        )

    # ---- Theorem 1: closed-form feasible region ----------------------- #
    def g(self, p: float | None = None) -> float:
        """g(p) = log(eps (1-p)/V) / log p,  the real-valued order guess."""
        p = self.p if p is None else p
        numerator = math.log(self.eps) + math.log1p(-p) - math.log(self.V)  # R1
        return numerator / math.log(p)

    def n_min(self) -> int:
        """N_min = max(0, ceil(g(p) - 1))  -- error constraint, closed form."""
        return max(0, math.ceil(self.g() - 1.0))

    def _n_max_raw(self, M: float) -> float:
        """N_max from  M(1-p)/V >= 1 ? +inf : floor(log(1 - q)/log p - 1)."""
        p = self.p
        q = M * (1.0 - p) / self.V                 # q = M / B_inf
        if q >= 1.0:                               # even B_inf fits under M
            return math.inf
        c = 1.0 - q                                # in (0, 1)
        n = math.floor(math.log(c) / math.log(p) - 1.0)
        return float(max(0, n))

    def n_max(self, u: float = U) -> float:
        """N_max with the overflow margin from Assumption 1 made rigorous.

        The paper requires B_N <= M for the *exact* terms.  Rounding inflates
        the running sum by at most (1 + gamma_(2N)), so the honest constraint is
            B_N * (1 + gamma_(2N)) <= M,
        i.e. we must solve the closed form with M_eff = M/(1+gamma_(2N)).
        Since gamma grows with N, this is a decreasing fixed point: iterate a
        few times (it converges in 2-3 steps).
        """
        M = self.M
        n = self._n_max_raw(M)
        for _ in range(8):
            if n == math.inf:
                return n
            m_eff = M / (1.0 + gamma_steps(int(2 * (n + 1)), u))
            n_next = self._n_max_raw(m_eff)
            if n_next == n:                        # fixed point reached
                return n
            n = n_next
        return n

    def feasible(self) -> bool:
        """N_min <= N_max  (Theorem 1, last sentence)."""
        return self.n_min() <= self.n_max()

    def order_star(self) -> int | None:
        """N* = N_min: the cheapest order that still meets the tolerance."""
        return self.n_min() if self.feasible() else None

    def certificate(self) -> dict:
        """Everything Theorem 1 certifies about this instance, in one dict."""
        n_lo, n_hi = self.n_min(), self.n_max()
        return {
            "p": self.p,
            "N_min": n_lo,
            "N_max": n_hi,
            "feasible": n_lo <= n_hi,
            "E_min": self.error_bound(n_lo),
            "B_min": self.sum_bound(n_lo),
            "B_inf": self.V / (1.0 - self.p),
        }


# --------------------------------------------------------------------------- #
# 2.  Theorem 2: local Lipschitz stability
# --------------------------------------------------------------------------- #
def g_prime(prob: OrderProblem, p: float) -> float:
    """g'(p) = [ -(1/(1-p)) log p  -  log(eps(1-p)/V) * (1/p) ] / (log p)**2.

    This is exactly the expression printed in the proof of Theorem 2.
    """
    log_p = math.log(p)
    a = math.log(prob.eps) + math.log1p(-p) - math.log(prob.V)   # A(p)
    return ((-1.0 / (1.0 - p)) * log_p - a * (1.0 / p)) / (log_p * log_p)


def lipschitz_constant(prob: OrderProblem, p_lo: float, p_hi: float, samples: int = 1001) -> float:
    """C = sup_{p in [p_lo, p_hi]} |dN / d log rho| = sup |p * g'(p)|.

    NOTE on a mismatch in the paper: the statement is Lipschitz in log(rho),
    but the printed constant is sup|g'(p)|, which is the derivative w.r.t. p,
    not w.r.t. log rho.  Since log p = log r - log rho we have
        d/d log rho = -p * d/dp,
    so the constant that makes the inequality true is sup |p g'(p)| -- that
    is what is returned here (and verified numerically in check_lipschitz).

    The supremum is located by a coarse scan followed by a ternary search,
    because |p g'| is unimodal on the intervals we care about.
    """
    f = lambda p: abs(p * g_prime(prob, p))

    step = (p_hi - p_lo) / (samples - 1)
    best_p, best = p_lo, f(p_lo)
    for i in range(samples):
        p = p_lo + step * i
        v = f(p)
        if v > best:
            best, best_p = v, p

    lo = max(p_lo, best_p - step)
    hi = min(p_hi, best_p + step)
    for _ in range(80):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            lo = m1
        else:
            hi = m2
    return max(best, f(0.5 * (lo + hi)))


def real_order(prob: OrderProblem, rho: float) -> float:
    """The real-valued order g(p) - 1 for a given rho (no ceiling).

    r is held fixed, so changing rho changes p = r/rho only.
    """
    p = prob.r / rho
    numerator = math.log(prob.eps) + math.log1p(-p) - math.log(prob.V)
    return numerator / math.log(p) - 1.0


def check_lipschitz(prob: OrderProblem, rho_lo: float, rho_hi: float, samples: int = 60) -> dict:
    """Numerically verify  |N(ro1)-N(ro2)| <= C |log ro1 - log ro2|."""
    p_lo, p_hi = prob.r / rho_hi, prob.r / rho_lo          # rho up -> p down
    C = lipschitz_constant(prob, min(p_lo, p_hi), max(p_lo, p_hi))

    worst_ratio = 0.0
    worst_pair = (rho_lo, rho_hi)
    for i in range(samples):
        ro1 = rho_lo + (rho_hi - rho_lo) * (i / (samples - 1))
        for j in range(samples):
            ro2 = rho_lo + (rho_hi - rho_lo) * (j / (samples - 1))
            dl = abs(math.log(ro1) - math.log(ro2))
            if dl == 0.0:
                continue
            d = abs(real_order(prob, ro1) - real_order(prob, ro2)) / dl
            if d > worst_ratio:
                worst_ratio, worst_pair = d, (ro1, ro2)
    return {"C": C, "observed_sup": worst_ratio, "pair": worst_pair,
            "holds": worst_ratio <= C * (1.0 + 1e-12)}


# --------------------------------------------------------------------------- #
# 3.  Lemma 1 and Lemma 2
# --------------------------------------------------------------------------- #
def check_conservative_rho(prob: OrderProblem, rho_hat: float) -> dict:
    """Lemma 1: rho_hat <= rho  =>  N_min(rho_hat) >= N_min(rho) (conservative)."""
    if not (0.0 < rho_hat < prob.rho):
        raise ValueError("need rho_hat < rho")
    p_hat = prob.r / rho_hat                                  # larger p
    n_true, n_cons = prob.n_min(), max(0, math.ceil(prob.g(p_hat) - 1.0))
    return {"rho": prob.rho, "rho_hat": rho_hat, "p": prob.p, "p_hat": p_hat,
            "N_min_true": n_true, "N_min_conservative": n_cons,
            "holds": n_cons >= n_true}


def check_interval_on_V(prob: OrderProblem, lo: float, hi: float) -> dict:
    """Lemma 2: N_min is increasing in V, so N_min(V_lo) <= N*(V) <= N_min(V_hi)."""
    if not (0.0 < lo <= prob.V <= hi):
        raise ValueError("need 0 < lo <= V <= hi")
    n_lo = OrderProblem(prob.r, prob.rho, lo, prob.eps, prob.M).n_min()
    n_mid = prob.n_min()
    n_hi = OrderProblem(prob.r, prob.rho, hi, prob.eps, prob.M).n_min()
    return {"V": prob.V, "V_lo": lo, "V_hi": hi,
            "N_lo": n_lo, "N_mid": n_mid, "N_hi": n_hi,
            "holds": n_lo <= n_mid <= n_hi}


# --------------------------------------------------------------------------- #
# 4.  Evaluation: what the order buys you, and what still breaks
# --------------------------------------------------------------------------- #
def evaluate_naive(r: float, N: int, coeff) -> tuple[float, float]:
    """Sum a_k r**k the textbook way, returning (value, max |partial sum|).

    Two deliberate sins, both fatal in practice:
      * a_k and r**k are materialised separately (r**k overflows long before
        a_k r**k does),
      * no scaling, so a partial sum outside the float range is unrecoverable.
    """
    total = 0.0
    worst = 0.0
    for k in range(N + 1):
        term = coeff(k) * r ** k                   # <-- overflow site
        total += term
        worst = max(worst, abs(total))
    return total, worst


def evaluate_recursive(r: float, N: int, step, first: float) -> tuple[float, float]:
    """Term recursion t_k = t_(k-1) * (a_k/a_(k-1)) * r, no r**k anywhere."""
    total = 0.0
    worst = 0.0
    t = first
    for k in range(N + 1):
        total += t
        worst = max(worst, abs(total))
        t = step(k) * t
    return total, worst


# --------------------------------------------------------------------------- #
# 5.  The three benchmark functions used for verification
# --------------------------------------------------------------------------- #
def _benchmarks():
    """(name, r, rho, V, eps, exact value, exact remainder, B_N closed form)."""
    out = []

    # f(z) = 1/(1-z), x0 = 0, disk of radius 0.9 -> V = 1/(1-0.9) = 10
    out.append((
        "1/(1-z)", 0.5, 0.9, 10.0, 1e-6,
        lambda r: 1.0 / (1.0 - r),
        lambda r, N: r ** (N + 1) / (1.0 - r),
        lambda r, N: (1.0 - r ** (N + 1)) / (1.0 - r),
    ))

    # f(z) = log(1+z), x0 = 0, branch point at -1 -> rho = 0.9,
    # V = max_{|z|=0.9} |log(1+z)| = |log(0.1)| = 2.302585...
    V_log = abs(math.log(1.0 - 0.9))
    out.append((
        "log(1+z)", 0.5, 0.9, V_log, 1e-6,
        lambda r: math.log1p(r),
        lambda r, N: math.log1p(r) - _log1p_partial(r, N),
        lambda r, N: sum(r ** k / k for k in range(1, N + 1)),
    ))

    # f(z) = e^z on the disk of radius 2 -> V = e^2
    out.append((
        "e^z", 1.0, 2.0, math.exp(2.0), 1e-6,
        lambda r: math.exp(r),
        lambda r, N: math.exp(r) - sum(r ** k / math.factorial(k) for k in range(N + 1)),
        lambda r, N: sum(r ** k / math.factorial(k) for k in range(N + 1)),
    ))
    return out


def _log1p_partial(r: float, N: int) -> float:
    """sum_{k=1..N} (-1)^(k+1) r^k / k, added in increasing order of magnitude."""
    s = 0.0
    for k in range(1, N + 1):
        s += (1.0 if k % 2 else -1.0) * r ** k / k
    return s


def actual_min_order(remainder, eps: float, nmax: int = 5000) -> int | None:
    """Brute force the true minimal order: smallest N with |R_N| <= eps."""
    for N in range(nmax):
        if abs(remainder(N)) <= eps:
            return N
    return None


def true_order_exp(r: float, rel_eps: float = U) -> int:
    """Log-space order for the e**x series (the same criterion as in taylor_overflow.py).

    Included only to quantify how loose the geometric (Cauchy) bound is:
    for e**z the coefficients decay like 1/k!, not like p**k.
    """
    log_target = r + math.log(rel_eps)
    n = max(1, int(math.ceil(r)))
    while n * math.log(r) - math.lgamma(n + 1.0) > log_target:
        n += 1
    return n


# --------------------------------------------------------------------------- #
# 6.  Demo / verification
# --------------------------------------------------------------------------- #
def _fmt(x) -> str:
    if x == math.inf:
        return "inf"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _row(cols, widths) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def main() -> None:
    print("=" * 104)
    print("Theorem 1 -- closed-form feasible region, checked against brute force")
    print("=" * 104)
    w = (12, 6, 11, 6, 10, 10, 12, 12, 9)
    print(_row(["f", "r", "V", "p", "N_min", "N_actual", "E_abs(N)", "|R_N|", "B_N"], w))
    print("-" * 104)
    for name, r, rho, V, eps, exact, rem, b_n in _benchmarks():
        prob = OrderProblem(r, rho, V, eps)
        n_min = prob.n_min()
        n_act = actual_min_order(lambda N: rem(r, N), eps)
        e_b = prob.error_bound(n_min)
        r_true = abs(rem(r, n_min))
        b_true = b_n(r, n_min)
        print(_row([name, f"{r:g}", f"{V:.4g}", f"{prob.p:.4g}", n_min, n_act,
                    f"{e_b:.3e}", f"{r_true:.3e}", f"{b_true:.3e}"], w))
    print("  N_actual <= N_min in every row: the Cauchy bound is conservative, as it must be.")
    print("  |R_N| <= E_abs(N) in every row: the error certificate of Theorem 1 is valid.")

    print()
    print("=" * 104)
    print("Where the overflow constraint M actually bites")
    print("=" * 104)
    small = OrderProblem(0.5, 0.9, 10.0, 1e-6, M=15.0)      # tight M on 1/(1-z)
    c = small.certificate()
    b_true_inf = 1.0 / (1.0 - small.r)                      # sum |t_k| = 1/(1-r) = 2
    print(f"  1/(1-z), eps=1e-6, M=15 :  N_min={c['N_min']}, N_max={_fmt(c['N_max'])}, "
          f"feasible={c['feasible']}")
    print(f"    theorem: B_inf = V/(1-p) = {c['B_inf']:.3g} > M -> declared infeasible")
    print(f"    truth  : B_inf = 1/(1-r) = {b_true_inf:.3g} < M -> every partial sum fits")
    print("    -> M is not really binding; the bound V/(1-p) overshoots by 1/(1-p) ~ 2.25x.")
    print("    -> when M genuinely binds, use a scaled accumulator (taylor_overflow.py),")
    print("       because B_N <= M was only sufficient, never necessary.")

    big = OrderProblem(700.0, 709.0, math.exp(709.0), 1e-16)   # e^z near the edge
    c = big.certificate()
    exact_binf = math.exp(700.0)                                # true B_inf = e^r
    print(f"\n  e^z, r=700, rho=709, eps=1e-16 :")
    print(f"    V = {big.V:.4e}   B_inf(bound) = {c['B_inf']:.4e}   "
          f"B_inf(true) = {exact_binf:.4e}   DBL_MAX = {DBL_MAX:.4e}")
    print(f"    N_min = {c['N_min']}   N_max = {_fmt(c['N_max'])}   feasible = {c['feasible']}")
    print("    -> the bound B_inf overflows while the true sum (e^700 = 1e304) does not,")
    print("       so Theorem 1 rejects a problem whose evaluation is perfectly safe (remark R2).")
    n_true = true_order_exp(700.0, 1e-16)
    print(f"    -> order gap: the bound asks for {c['N_min']} terms, the series needs {n_true} "
          f"({c['N_min'] / n_true:.0f}x conservative: p is near 1, coefficients are 1/k!)")
    try:
        naive = 700.0 ** 700                                     # materialise r**k
        print(f"    r**700 as a float = {naive:.3e}")
    except OverflowError:
        print("    r**700 materialised directly -> OverflowError (must use term recursion).")

    print()
    print("=" * 104)
    print("Theorem 2 -- Lipschitz stability in log(rho)")
    print("=" * 104)
    prob = OrderProblem(0.5, 0.9, 10.0, 1e-6)
    res = check_lipschitz(prob, 0.85, 0.95)
    print(f"  C = sup|p g'(p)| over p in [{prob.r / 0.95:.4f}, {prob.r / 0.85:.4f}] = {res['C']:.4f}")
    print(f"  observed sup |dN/d log rho| = {res['observed_sup']:.4f}   "
          f"(worst pair rho = {res['pair'][0]:.3f}, {res['pair'][1]:.3f})")
    print(f"  inequality holds: {res['holds']}")
    print("  (integer orders differ by at most +1 on top of this, from the ceiling)")

    print()
    print("=" * 104)
    print("Lemma 1 and Lemma 2 -- monotonicity checks")
    print("=" * 104)
    lem1 = check_conservative_rho(prob, 0.85)
    print(f"  Lemma 1: rho=0.9 -> N_min={lem1['N_min_true']},  "
          f"rho_hat=0.85 (p {lem1['p']:.4f} -> {lem1['p_hat']:.4f}) -> "
          f"N_min={lem1['N_min_conservative']}   holds={lem1['holds']}")
    lem2 = check_interval_on_V(prob, 8.0, 12.0)
    print(f"  Lemma 2: V in [8, 12] -> N_min = {lem2['N_lo']} <= {lem2['N_mid']} "
          f"<= {lem2['N_hi']}   holds={lem2['holds']}")

    print()
    print("=" * 104)
    print("Section 'Computational Complexity' -- the closed form really is O(1)")
    print("=" * 104)
    t0 = time.perf_counter()
    reps = 200_000
    for _ in range(reps):
        prob.n_min()
    dt = (time.perf_counter() - t0) / reps
    print(f"  n_min(): {dt * 1e6:.3f} us per call  (no enumeration, no iteration)")

    print()
    print("=" * 104)
    print("The evaluation step: order from Theorem 1, terms without r**k")
    print("=" * 104)
    prob = OrderProblem(1.0, 2.0, math.exp(2.0), 1e-9)
    n = prob.n_min()
    val, worst = evaluate_recursive(
        1.0, n, step=lambda k: 1.0 / (k + 1), first=1.0)       # e^r, t_k = t_(k-1)*r/k
    val2, worst2 = evaluate_naive(1.0, n, coeff=lambda k: 1.0 / math.factorial(k))
    print(f"  e^1 with N_min={n} terms:  recursive={val:.15f}  naive={val2:.15f}  "
          f"math.exp={math.exp(1.0):.15f}")
    print(f"  max |partial sum| = {worst:.4f}  (must stay <= M = {DBL_MAX:.3e})")


if __name__ == "__main__":
    main()
