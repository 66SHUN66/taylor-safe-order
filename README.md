# taylor-safe-order

Automatic order selection for Taylor expansions that is **as accurate as the
floating-point format allows** and **never overflows** on the way there.

Zero third-party dependencies. Pure `math` + `dataclasses`.

```python
from taylor_auto import scan, evaluate_scaled, exp_terms

r = scan(exp_terms(10.0))                       # which order is best?
v = evaluate_scaled(exp_terms(10.0), r.n_star)  # evaluate it safely
print(r.n_star, v, r.verdict())                 # 45  22026.465794807  OK
```

---

## Why "more terms" is not "more accurate"

Two error curves cross. Truncation error shrinks with the order, but the
rounding error **grows** with it, because every added term costs about half an
ulp of the running sum:

| | grows with N | shrinks with N |
|---|---|---|
| truncation `R_N` | | ✓ |
| rounding `~ (N+1)·u·Σ\|t_k\|` | ✓ | |

So the order that gives the *most accurate* double is an interior minimum:

```
est(N) = |t_{N+1}| / (1 - r)  +  (N+1)·u·B_N       r = |t_{N+1}/t_N|,  B_N = Σ|t_k|
n*     = argmin_N est(N)      subject to  N <= N_overflow
```

This is the classical **optimal truncation** of an asymptotic series; for a
convergent one it lands on the point where the term drops below half an ulp of
the sum. Nothing here is a heuristic — `est` is a bound for alternating series
(Leibniz) and a geometric-tail estimate otherwise, and the repo's tests show
the detected `n*` coinciding with the brute-force argmin.

## Three failure modes, three tools

| file | question it answers |
|---|---|
| `taylor_auto.py` | *Given a series:* which order is most accurate, and where does the range end? |
| `taylor_order.py` | *Given `ρ, V, ε`:* the closed-form `N_min / N_max` from the Cauchy bound, with proofs. |
| `taylor_overflow.py` | *Why does it break,* and how do argument reduction + scaled summation fix it? |
| `scaled_sum.py` | The `(mantissa, exponent)` accumulator shared by both evaluators. |

Naive Taylor evaluation dies in three different ways, and each needs its own fix:

1. **The term overflows** — `r**k` leaves the double range long before `r**k/k!`
   does (`700**700 → OverflowError`, although the term is `1e-1000`).
   → generate terms **recursively**, `t_k = t_{k-1} * r / k` (divide first).
2. **The partial sum overflows** — terms pass through a peak of `1e306` even
   when the answer is `1e-300`.
   → accumulate in `ScaledSum`, keeping `m * 2**e` separate.
3. **Cancellation** — `e**-x` through its own Taylor series loses ~600 digits.
   → do not sum it; use `1/e**x` (`exp_reduced`), i.e. reduce the argument.

## What it actually measures

`python taylor_auto.py`

```
x        N*       plateau   est rel err   rel err vs libm   n_overflow   verdict
1        17       18        2.06e-15      1.63e-16          none         OK
10       45       46        5.21e-15      1.65e-16          none         OK
100      191      192       2.15e-14      1.84e-16          none         OK
700      923      924       1.03e-13      2.16e-15          none         OK
709      933      934       1.04e-13      8.50e-16          none         OK
710      709      none      3.02e-02      n/a              733          FAIL: range ends at term 733
800      457      none      7.52e-01      n/a              458          FAIL: range ends at term 458
```

- `N* = plateau - 1` for every convergent case: the detected optimum sits exactly
  one term before the sum reaches its noise floor.
- `x = 710`: the true value `2.3e308` does not fit in a double, so the honest
  answer is `FAIL`, not `inf`.
- The estimate is an **upper bound**, not a prediction: `1.03e-13` at `x = 700`
  against an achieved `2.16e-15` (≈50× conservative, by design).

Divergent series is where the detector earns its keep —
`I(x) = ∫₀^∞ e^{-t}/(1+xt) dt ~ Σ (-1)^k k! x^k`, `x = 0.1`, reference from
Simpson quadrature:

```
k=8   rel err 2.03e-04
k=9   rel err 1.93e-04   <- N* (and the brute-force argmin)
k=10  rel err 2.03e-04
k=15  rel err 8.66e-04   <- more terms, worse answer
```

Plain Taylor `sin x` shows the opposite extreme — at large `x` **no** order helps,
because the peak term `1e42` costs ~26 digits of rounding:

```
x=10   N*=46  err 8.2e-13
x=20   N*=66  err 4.3e-08
x=50   FAIL  peak term 2.9e20  -> est. relative error >= 1
x=100  FAIL  peak term 1.1e42  -> est. relative error >= 1
```

That is the mathematical justification for range reduction: reduce `x` to
`|r| ≤ π/4` and 13 terms give `1e-16` (`sin_cos`).

## The theorem side (`taylor_order.py`)

`f` analytic in `D(x₀, ρ)`, `p = r/ρ`, `V = sup|f|` on the disk. Then

```
|R_N| <= V/(1-p) · p^(N+1)                    (truncation certificate)
B_N   =  V · Σ_{k<=N} p^k  <= M               (overflow certificate)

N_min = max(0, ceil(log(ε(1-p)/V)/log p - 1))
N_max = inf  if M(1-p)/V >= 1  else  floor(log(1 - M(1-p)/V)/log p - 1)
feasible  <=>  N_min <= N_max,        N* = N_min
```

Verified against brute force in `python taylor_order.py`:

| f | ρ | V | N_min | N_actual | `E_abs(N_min)` | `\|R_N\|` |
|---|---|---|---|---|---|---|
| `1/(1-z)` | 0.9 | 10 | 28 | 20 | 8.90e-07 | 3.73e-09 |
| `log(1+z)` | 0.9 | 2.303 | 26 | 15 | 6.64e-07 | 1.86e-10 |
| `e^z` | 2.0 | 7.389 | 23 | 9 | 8.81e-07 | 0 |

`N_actual <= N_min` in every row (the Cauchy bound is conservative, as it must
be) and `|R_N| <= E_abs` in every row (the certificate is valid).

Three things worth knowing about the bound itself:

- **It must be evaluated in log space.** For `ρ = 709, ε = 1e-16` the quotient
  `ε(1-p)/V ≈ 1.5e-326` is below the smallest subnormal, so a literal
  `math.log(eps*(1-p)/V)` raises `ValueError`; the code uses
  `log(ε) + log1p(-p) - log(V)`.
- **`B_N` is wildly conservative.** For `e^z` at `r = 700` the bound demands
  *58 723* terms while the series needs 924, and `B_∞` overflows even though the
  true sum (`1e304`) does not — so Theorem 1 can reject a perfectly evaluable
  problem. `certificate()` reports both the bound and the truth.
- **The Lipschitz constant needs `p`, not `g'`.** The statement is Lipschitz in
  `log ρ`, and since `d/d log ρ = -p·d/dp` the correct constant is
  `C = sup|p·g'(p)|` (63.09 measured over `p ∈ [0.526, 0.588]`, observed ratio
  62.85). Using the printed `sup|g'(p)|` misses a factor `p ≈ 0.55`.

## API

```python
from taylor_auto import scan, Result, evaluate_plain, evaluate_scaled
from taylor_auto import exp_terms, sin_terms, asymptotic_euler_terms
from scaled_sum import ScaledSum
from taylor_order import OrderProblem, lipschitz_constant, check_conservative_rho
from taylor_overflow import exp_reduced, sin_cos, exp_order, peak_log_term_exp
```

`scan(terms, u=U, max_terms=1e6, patience=100, plateau_guard=16) -> Result`
with fields `n_star, value, est_rel, est_abs, n_overflow, n_plateau, peak_term,
last_k`, plus `verdict()` (`OK` / `WEAK` / `FAIL: ...`) and `usable`.

### Plugging in your own expansion

Write **one generator** that yields `t_0, t_1, …` recursively — never
`a_k * x**k`, because that product is what overflows:

```python
def my_terms(x):
    t, k = 1.0, 0                 # t_0
    while True:
        yield t
        k += 1
        t = t * x / k             # t_k from t_(k-1); divide before multiply
```

## Performance

`scan` on the `e^700` series (940 terms visited): **410 µs** (~2.3 M terms/s).
`ScaledSum.add`: **250 ns** per term. `OrderProblem.n_min()`: **0.26 µs**,
`O(1)` with no enumeration.

## Limitations

- 53 bits of precision only. For more, feed the same terms to `Decimal` /
  `mpmath` — the detector's stopping rules still apply.
- `sin_cos` reduces with a 2-word split of `π/2`: exact to `|x| ≲ 1e7`, degrades
  to `~1e-9` at `1e8`, `~1e-7` at `1e10`. Beyond that use Payne–Hanek (`math.sin`).
- The closed form assumes a Cauchy-bounded disk; it says nothing about an
  arbitrary `f` with no known `ρ, V`.

## Layout

```
scaled_sum.py        overflow-proof accumulator
taylor_order.py      closed-form order bounds (theorem + numerical checks)
taylor_auto.py       order detector + safe evaluators
taylor_overflow.py   argument reduction, e^x / sin, failure modes
tests/test_accuracy.py
```

Run everything:

```powershell
python taylor_auto.py          # detector demo
python taylor_order.py         # theorem verification tables
python taylor_overflow.py      # reduction vs naive Taylor
python tests\test_accuracy.py  # assertions (also pytest-compatible)
```

## License

MIT — see `LICENSE`.
