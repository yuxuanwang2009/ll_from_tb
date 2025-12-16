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

    # Square-lattice nearest-neighbor tight-binding:
    #   E(k) = -2t (cos kx + cos ky)
    # Near k=0: E ≈ -4t + t (kx^2 + ky^2) = -4t + k^2/(2m*)
    # so m* = 1/(2t) in units ħ=a=1.
    t = 0.7
    m_star = 1.0 / (2.0 * t)

    hops = [
        Hopping2D(dx=0, dy=0, mat=np.array([[0.0]])),
        Hopping2D(dx=1, dy=0, mat=np.array([[-t]])),
        Hopping2D(dx=0, dy=1, mat=np.array([[-t]])),
    ]

    # Choose mu slightly above the band bottom (-4t) so the FS is a single small pocket at k=0.
    epsF = 0.08 * t
    mu = -4.0 * t + epsF

    # Use fluxes phi = 1/q (weak field = large q).
    qs = [53, 61, 71, 83, 97, 113]
    fluxes = [(1, q) for q in qs]
    print("=== Peierls substitution + exact diagonalization demo ===")
    print(f"t                        = {t:.6g}")
    print(f"m* (parabolic approx)     = {m_star:.6g}  (=1/(2t))")

    # Plot microscopic Landau-like levels near μ as dots vs B.
    levels_path = str(Path(__file__).with_name("ll_levels_dots_vs_B_peierls_tb.png"))
    dots = plot_levels_window_vs_B_ed(
        hops=hops,
        fluxes=fluxes,
        mu=mu,
        window=0.25,
        outpath=levels_path,
        nkx=7,
        nky=7,
        title="Peierls + exact diag: levels near μ (dots) vs B (square-lattice TB)",
    )
    print(f"saved levels plot          = {levels_path}")

    # Spacing plot computed from the dots above (no extra diagonalization).
    spacing_path = str(Path(__file__).with_name("ll_spacing_vs_B_peierls_tb.png"))
    expected_slope = 1.0 / m_star
    out = plot_ll_spacing_vs_B_from_levels_window(
        B=dots["B"],
        E=dots["E"],
        mu=mu,
        outpath=spacing_path,
        navg=1,
        expected_slope=expected_slope,
        title="Peierls + exact diag: ΔE near μ vs B (from dots)",
        fit_through_origin=True,
    )
    print(f"expected slope (1/m*)     = {expected_slope:.6g}")
    print(f"fit slope d(ΔE)/dB        = {out['fit_slope']:.6g}  ({'through origin' if out['fit_through_origin'] else 'affine'})")
    print(f"saved spacing plot         = {spacing_path}")


if __name__ == "__main__":
    main()
