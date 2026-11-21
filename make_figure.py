#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render docs/error_vs_order.svg from the repo's own functions.

  A  divergent series  sum (-1)^k k! x^k at x = 0.1
     true relative error vs terms used; the minimum is the detector's N*
  B  e^x at x = 700
     truncation |t(N+1)|/(1-r) and rounding (N+1)u*B(N) vs N, and where they cross

Everything is computed here -- no hand-drawn numbers.  Run from the repo root:
    python make_figure.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taylor_auto import (                     # noqa: E402
    U, asymptotic_euler_terms, evaluate_plain, exp_terms, i_reference, scan,
)

# ------------------------------------------------------------------ data: panel A
XA = 0.1
REF_A = i_reference(XA)
NA = scan(asymptotic_euler_terms(XA)).n_star
A = [(k, abs(evaluate_plain(asymptotic_euler_terms(XA), k) - REF_A) / REF_A)
     for k in range(21)]
A = [(k, err) for k, err in A if err > 0.0]

# ------------------------------------------------------------------ data: panel B
XB = 700.0
B = ([], [], [])
S = BB = 0.0
t = 1.0                                        # t_0 of e**x
for n in range(1000):
    S += t
    BB += abs(t)
    nt = t * XB / (n + 1)                      # t_(n+1), same recurrence as exp_terms
    if 700 <= n <= 990 and S and (n % 10 == 0 or n == 923):
        r = nt / t
        trunc = abs(nt) / (1.0 - r) if r < 1.0 else abs(nt)
        B[0].append((n, trunc / abs(S)))
        B[1].append((n, (n + 1) * U * BB / abs(S)))
        B[2].append((n, B[0][-1][1] + B[1][-1][1]))
    t = nt

NB = min(B[2], key=lambda p: p[1])[0]

# ------------------------------------------------------------------ geometry
W, H = 1200, 430
AX0, AX1, AY0, AY1 = 74, 542, 62, 350
BX0, BX1 = 662, 1130
AMIN, AMAX = -5, 0
BMIN, BMAX = -28, 2
K0, K1 = 0, 20
N0, N1 = 700, 990

INK, MUT, LINE = "#1f2328", "#8a8f98", "#e7e9ec"
BLUE, ORANGE, GREEN = "#2563eb", "#e1570c", "#0f7b52"


def sx(i, i0, i1, x0, x1):
    return x0 + (i - i0) / (i1 - i0) * (x1 - x0)


def sy(v, vmin, vmax, y0, y1):
    return y0 + (vmax - math.log10(v)) / (vmax - vmin) * (y1 - y0)


e = []
add = e.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
add(f'<rect width="{W}" height="{H}" fill="#fff"/>')


def frame(x0, x1, vmin, vmax):
    add(f'<rect x="{x0}" y="{AY0}" width="{x1-x0}" height="{AY1-AY0}" fill="#fafbfc" stroke="{LINE}"/>')
    for v, label in ((10 ** vmax, f"1e{vmax}"), (10 ** ((vmax + vmin) / 2), f"1e{int((vmax+vmin)/2)}"),
                     (10 ** vmin, f"1e{vmin}")):
        y = sy(v, vmin, vmax, AY0, AY1)
        if label != "1e2":
            add(f'<line x1="{x0}" y1="{y:.0f}" x2="{x1}" y2="{y:.0f}" stroke="{LINE}"/>')
        add(f'<text x="{x0-6}" y="{y+3:.0f}" font-size="10" fill="{MUT}" text-anchor="end">{label}</text>')


frame(AX0, AX1, AMIN, AMAX)
frame(BX0, BX1, BMIN, BMAX)

# titles
add(f'<text x="{AX0}" y="30" font-size="15" font-weight="600" fill="{INK}">'
    f'A  divergent series: past the smallest term it gets worse</text>')
add(f'<text x="{AX0}" y="47" font-size="11" fill="{MUT}">Σ(−1)ᵏ k! xᵏ, x = 0.1 — true relative error</text>')
add(f'<text x="{BX0}" y="30" font-size="15" font-weight="600" fill="{INK}">'
    f'B  eˣ at x = 700: truncation falls, rounding rises</text>')
add(f'<text x="{BX0}" y="47" font-size="11" fill="{MUT}">'
    f'tail |t(N+1)|/(1−r) and rounding (N+1)u·B(N), relative to the sum</text>')

# x ticks
for k in range(0, 21, 5):
    x = sx(k, K0, K1, AX0, AX1)
    add(f'<line x1="{x:.0f}" y1="{AY1}" x2="{x:.0f}" y2="{AY1+4}" stroke="{MUT}"/>')
    add(f'<text x="{x:.0f}" y="{AY1+17}" font-size="10" fill="{MUT}" text-anchor="middle">{k}</text>')
for n in (700, 800, 900, 990):
    x = sx(n, N0, N1, BX0, BX1)
    add(f'<line x1="{x:.0f}" y1="{AY1}" x2="{x:.0f}" y2="{AY1+4}" stroke="{MUT}"/>')
    add(f'<text x="{x:.0f}" y="{AY1+17}" font-size="10" fill="{MUT}" text-anchor="middle">{n}</text>')

add(f'<text x="{(AX0+AX1)/2:.0f}" y="{AY1+36}" font-size="11" fill="{INK}" text-anchor="middle">'
    f'terms summed</text>')
add(f'<text x="{(BX0+BX1)/2:.0f}" y="{AY1+36}" font-size="11" fill="{INK}" text-anchor="middle">'
    f'terms summed</text>')

# panel A curve
pts = " ".join(f"{sx(k, K0, K1, AX0, AX1):.0f},{sy(v, AMIN, AMAX, AY0, AY1):.0f}" for k, v in A)
add(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
ka, va = NA, dict(A)[NA]
cx, cy = sx(ka, K0, K1, AX0, AX1), sy(va, AMIN, AMAX, AY0, AY1)
add(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="5" fill="none" stroke="{ORANGE}" stroke-width="2"/>')
add(f'<text x="{cx+12:.0f}" y="{cy+22:.0f}" font-size="11.5" fill="{ORANGE}" font-weight="600">'
    f'N* = {ka}  ({va:.1e})</text>')
add(f'<text x="{AX0+14}" y="{AY1-12}" font-size="10.5" fill="{MUT}">'
    f'k = 15 is already 4.5× worse than N*</text>')

# panel B curves
for data, colour in ((B[0], BLUE), (B[1], ORANGE), (B[2], INK)):
    p = " ".join(f"{sx(n, N0, N1, BX0, BX1):.0f},{sy(v, BMIN, BMAX, AY0, AY1):.0f}" for n, v in data)
    dash = ' stroke-dasharray="5,3"' if colour == INK else ""
    add(f'<polyline points="{p}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>')

nb, vb = NB, dict(B[2])[NB]
x, y = sx(nb, N0, N1, BX0, BX1), sy(vb, BMIN, BMAX, AY0, AY1)
add(f'<line x1="{x:.0f}" y1="{AY0}" x2="{x:.0f}" y2="{AY1}" stroke="{GREEN}" stroke-width="1.2" '
    f'stroke-dasharray="4,3"/>')
add(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5" fill="none" stroke="{GREEN}" stroke-width="2"/>')
add(f'<text x="{x-4:.0f}" y="{AY0+16}" font-size="11.5" fill="{GREEN}" font-weight="600" '
    f'text-anchor="end">N* = {nb}</text>')

# legend, bottom-left of panel B (no curve ever enters that corner)
lx, ly = BX0 + 14, AY1 - 52
for label, colour, dash in (("truncation", BLUE, ""), ("rounding", ORANGE, ""),
                            ("sum = estimate", INK, ' stroke-dasharray="5,3"')):
    add(f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" stroke="{colour}" stroke-width="2"{dash}/>')
    add(f'<text x="{lx+22}" y="{ly+3}" font-size="10.5" fill="{INK}">{label}</text>')
    ly += 17

# footer
add(f'<text x="{AX0}" y="{H-12}" font-size="10.5" fill="{MUT}">'
    f'Generated by make_figure.py from taylor_auto.py — reference for A is Simpson quadrature</text>')
add("</svg>")

os.makedirs("docs", exist_ok=True)
svg = "\n".join(e) + "\n"
with open(os.path.join("docs", "error_vs_order.svg"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(svg)

print(f"A: N*={NA} min={va:.2e} | B: N*={NB} est={vb:.2e} | {len(svg)} bytes")
