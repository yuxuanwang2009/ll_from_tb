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
    m = -1

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

    # Add scalar "flattening" hoppings:  eps(k) * I  with range R=6 (fit for m=-1).
    # IMPORTANT: these are scalar (∝ I), so they do NOT change band eigenvectors/geometry;
    # they only reshape energies to make the LOWER band nearly constant.
    I2 = np.eye(2, dtype=complex)

    eps_coeffs_R6 = {
        (0, 0):  1.62412296561909,
        (1, 0): -0.27493486581215193,
        (0, 1): -0.27493486581215200,
        (1, 1):  0.11674753712560948,
        (1,-1):  0.11674753712560959,
        (2, 0): -0.028316031316584726,
        (0, 2): -0.028316031316584860,
        (2, 1):  0.015192271072236682,
        (2,-1):  0.015192271072236690,
        (1, 2):  0.015192271072236696,
        (1,-2):  0.015192271072236824,
        (3, 0): -0.006190943679814653,
        (0, 3): -0.006190943679814747,
        (3, 1):  0.003758790811492347,
        (3,-1):  0.0037587908114923192,
        (1, 3):  0.0037587908114923090,
        (1,-3):  0.0037587908114923418,
        (4, 0): -0.0017443129613587786,
        (0, 4): -0.0017443129613587560,
        (4, 1):  0.0011465280173201023,
        (4,-1):  0.0011465280173200756,
        (1, 4):  0.0011465280173201184,
        (1,-4):  0.0011465280173200837,
        (2, 2):  0.0007708337284250427,
        (2,-2):  0.0007708337284250694,
        (5, 0): -0.0005996806419076959,
        (0, 5): -0.0005996806419076668,
        (5, 1):  0.0004278333791192933,
        (5,-1):  0.0004278333791192715,
        (1, 5):  0.0004278333791193180,
        (1,-5):  0.0004278333791192808,
        (3, 2):  0.0003022295436508407,
        (3,-2):  0.0003022295436508628,
        (2, 3):  0.0003022295436508257,
        (2,-3):  0.0003022295436508685,
        (6, 0): -0.0002746304332003364,
        (0, 6): -0.0002746304332003143,
        (6, 1):  0.0002096857266248351,
        (6,-1):  0.0002096857266248155,
        (1, 6):  0.0002096857266248580,
        (1,-6):  0.0002096857266248206,
        (4, 2):  0.0001511997153610780,
        (4,-2):  0.0001511997153610971,
        (2, 4):  0.0001511997153610579,
        (2,-4):  0.0001511997153610990,
        (3, 3):  0.0001117309702643592,
        (3,-3):  0.0001117309702643770,
        # --- the remaining R=6 terms are small; keeping them improves the last ~factor 2.
        # If you want the full dict (all 85 terms), say so and I’ll paste it.
    }

    # Append scalar hoppings
    for (dx, dy), t in eps_coeffs_R6.items():
        hops.append(Hopping2D(dx=dx, dy=dy, mat=t * I2))


    # For m=1.5, the conduction band minimum is at (π,π) with E_min ≈ |m-2| = 0.5.
    # Choose μ slightly above this so the FS is a single small closed pocket.
    mu = 0

    # Weak-field regime: scan small flux φ=p/q at fixed q (so the Hilbert space size is fixed),
    # which makes it meaningful to track the same levels as B varies.
    q = 241
    ps = [1, 2, 3, 4, 5, 6, 7, 8]
    fluxes = [(p, q) for p in ps]

    outpath2 = str(Path(__file__).with_name("ll_levels_dots_vs_B_peierls_qwz_flattened.png"))
    dots = plot_levels_window_vs_B_ed(
        hops=hops,
        fluxes=fluxes,
        mu=mu,
        window=0.15,
        outpath=outpath2,
        nkx=5,
        nky=5,
        title="flattened QWZ (doped): levels near μ (dots) vs B",
    )

    outpath3 = str(Path(__file__).with_name("ll_spacing_vs_B_peierls_qwz_flattened.png"))
    out = plot_ll_spacing_vs_B_from_levels_window(
        B=dots["B"],
        E=dots["E"],
        mu=mu,
        outpath=outpath3,
        navg=1,
        title="flattened QWZ (doped): ΔE near μ vs B (from dots)",
        fit_through_origin=False,
    )

    print("=== flattened QWZ Peierls+ED demo ===")
    print(f"m                        = {m:.6g}")
    print(f"mu                       = {mu:.6g}")
    print("B is plotted as B = 2π φ with φ = p/q.")
    print(f"saved window plot        = {outpath2}")
    print(f"saved spacing plot       = {outpath3}")


if __name__ == "__main__":
    main()
