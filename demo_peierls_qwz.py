from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ll_peierls_ed import (
    Hopping2D,
    plot_levels_window_vs_B_ed,
    plot_ll_spacing_vs_B_from_levels_window,
)


def main() -> None:
    mpldir = Path(__file__).with_name(".mplconfig")
    mpldir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpldir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpldir))

    # Qi-Wu-Zhang (QWZ) Chern insulator (2-band) on a square lattice:
    #   H(k) = sin(kx) σx + sin(ky) σy + (m + cos(kx) + cos(ky)) σz
    # (units: ħ=a=1)
    m = 1.5

    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    # Real-space hoppings matching the Bloch Hamiltonian above.
    # cos terms: (1/2)(e^{ik}+e^{-ik}) σz => hop (+x) contributes +0.5 σz (and h.c. adds -x).
    # sin terms: (1/2i)(e^{ik}-e^{-ik}) σ{ x,y } => hop (+x) contributes -(i/2) σx, etc.
    hops = [
        Hopping2D(dx=0, dy=0, mat=m * sz),
        Hopping2D(dx=1, dy=0, mat=0.5 * sz + (-0.5j) * sx),
        Hopping2D(dx=0, dy=1, mat=0.5 * sz + (-0.5j) * sy),
    ]

    # For m=1.5, the conduction band minimum is at (π,π) with E_min ≈ |m-2| = 0.5.
    # Choose μ slightly above this so the FS is a single small closed pocket.
    mu = 0.65

    # Weak-field regime: scan small flux φ=p/q at fixed q (so the Hilbert space size is fixed),
    # which makes it meaningful to track the same levels as B varies.
    q = 241
    ps = [1, 2, 3, 4, 5, 6, 7, 8]
    fluxes = [(p, q) for p in ps]

    outpath2 = str(Path(__file__).with_name("ll_levels_dots_vs_B_peierls_qwz.png"))
    dots = plot_levels_window_vs_B_ed(
        hops=hops,
        fluxes=fluxes,
        mu=mu,
        window=0.15,
        outpath=outpath2,
        nkx=5,
        nky=5,
        title="QWZ (doped): levels near μ (dots) vs B",
    )

    outpath3 = str(Path(__file__).with_name("ll_spacing_vs_B_peierls_qwz.png"))
    out = plot_ll_spacing_vs_B_from_levels_window(
        B=dots["B"],
        E=dots["E"],
        mu=mu,
        outpath=outpath3,
        navg=1,
        title="QWZ (doped): ΔE near μ vs B (from dots)",
        fit_through_origin=False,
    )

    print("=== QWZ Peierls+ED demo ===")
    print(f"m                        = {m:.6g}")
    print(f"mu                       = {mu:.6g}")
    print("B is plotted as B = 2π φ with φ = p/q.")
    print(f"saved window plot        = {outpath2}")
    print(f"saved spacing plot       = {outpath3}")
    print(f"fit slope d(ΔE)/dB        = {out['fit_slope']:.6g}  ({'through origin' if out['fit_through_origin'] else 'affine'})")


if __name__ == "__main__":
    main()
