from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Hopping2D:
    """
    Real-space hopping term for a 2D square lattice model.

    The model is specified on lattice sites r=(x,y) with integer coordinates and
    `norb` orbitals per site. A hopping contributes

        c†_{r+R,β}  t_{β,α}(R)  c_{r,α}  + h.c.

    where R=(dx,dy) and `mat` is the (norb,norb) matrix in orbital space.
    """

    dx: int
    dy: int
    mat: ArrayLike  # shape (norb, norb)


def _combine_hoppings(hops: Iterable[Hopping2D]) -> dict[tuple[int, int], ArrayLike]:
    combined: dict[tuple[int, int], ArrayLike] = {}
    for hop in hops:
        key = (int(hop.dx), int(hop.dy))
        mat = np.asarray(hop.mat)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("Each hopping mat must be a square (norb,norb) matrix.")
        if key in combined:
            combined[key] = combined[key] + mat
        else:
            combined[key] = mat.astype(complex, copy=True)
    return combined


def symmetrize_hoppings(hops: Iterable[Hopping2D]) -> list[Hopping2D]:
    """
    Ensure the hopping list defines a Hermitian model by adding missing reverse hops.
    """
    combined = _combine_hoppings(hops)
    out = dict(combined)
    for (dx, dy), mat in list(combined.items()):
        rkey = (-dx, -dy)
        rmat = mat.conj().T
        if rkey in out:
            # user may already have provided both; still enforce Hermiticity by averaging
            out[rkey] = 0.5 * (out[rkey] + rmat)
            out[(dx, dy)] = 0.5 * (out[(dx, dy)] + out[rkey].conj().T)
        else:
            out[rkey] = rmat
    return [Hopping2D(dx=k[0], dy=k[1], mat=v) for k, v in out.items()]


def peierls_phase_landau_gauge(
    *,
    phi: float,
    x: int,
    dx: int,
    dy: int,
) -> complex:
    """
    Peierls phase for a hop (x,y)->(x+dx,y+dy) in Landau gauge A=(0,Bx).

    Uses straight-line integral; with lattice constant a=1, the phase is:

        exp(i 2π φ * (x + dx/2) * dy)

    where φ is flux per plaquette in units of the flux quantum (Φ0).
    """
    return np.exp(1j * 2.0 * np.pi * phi * (float(x) + 0.5 * float(dx)) * float(dy))


def magnetic_bloch_hamiltonian(
    *,
    kx: float,
    ky: float,
    hops: Iterable[Hopping2D],
    p: int,
    q: int,
) -> ArrayLike:
    """
    Magnetic Bloch Hamiltonian (Hofstadter) for flux φ=p/q on a square lattice.

    Gauge choice: Landau gauge A=(0,Bx). The magnetic unit cell is q sites in x.

    Momentum ranges:
      - kx in [-π/q, π/q)
      - ky in [-π, π)

    Returns an array of shape (q*norb, q*norb).
    """
    if q <= 0:
        raise ValueError("q must be positive.")
    p = int(p)
    q = int(q)
    phi = p / q

    hops_sym = symmetrize_hoppings(hops)
    combined = _combine_hoppings(hops_sym)
    any_mat = next(iter(combined.values()))
    norb = int(any_mat.shape[0])

    H = np.zeros((q * norb, q * norb), dtype=complex)

    for m in range(q):
        row_base = m * norb
        for (dx, dy), mat in combined.items():
            m2 = (m + dx) % q
            shift = (m + dx - m2) // q  # integer cell shift in x by q*shift
            col_base = m2 * norb

            phase = peierls_phase_landau_gauge(phi=phi, x=m, dx=dx, dy=dy)
            bloch = np.exp(1j * (ky * float(dy) + kx * float(q * shift)))

            H[row_base : row_base + norb, col_base : col_base + norb] += bloch * phase * mat

    # Numerical symmetrization (helps tiny drift from float ops).
    return 0.5 * (H + H.conj().T)


def hofstadter_bands(
    *,
    hops: Iterable[Hopping2D],
    p: int,
    q: int,
    nkx: int = 9,
    nky: int = 9,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute magnetic subband energies on a uniform k-grid in the magnetic BZ.

    Returns (kx_grid, ky_grid, evals) where evals has shape (Nk, q*norb),
    with Nk = nkx*nky and rows corresponding to k-points in row-major order.
    """
    if nkx <= 0 or nky <= 0:
        raise ValueError("nkx and nky must be positive.")
    kx_grid = np.linspace(-np.pi / q, np.pi / q, nkx, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)

    evals_list: list[ArrayLike] = []
    for kx in kx_grid:
        for ky in ky_grid:
            Hk = magnetic_bloch_hamiltonian(kx=float(kx), ky=float(ky), hops=hops, p=p, q=q)
            evals_list.append(np.linalg.eigvalsh(Hk))
    evals = np.asarray(evals_list, dtype=float)
    return kx_grid, ky_grid, evals


def extract_level_energies_from_bands(evals: ArrayLike) -> dict[str, ArrayLike]:
    """
    Reduce dispersive magnetic subbands to "level energies" via robust statistics.

    For each band index j, compute:
      - E_med[j] = median over k
      - width[j] = max-min over k
    """
    evals = np.asarray(evals, dtype=float)
    if evals.ndim != 2:
        raise ValueError("evals must have shape (Nk, nband).")
    E_med = np.median(evals, axis=0)
    width = np.max(evals, axis=0) - np.min(evals, axis=0)

    # Subbands can cross as a function of k, so the band index is not a robust label.
    # For “LL-like” uses near a fixed μ, sorting by the representative energy is the
    # most stable choice.
    order = np.argsort(E_med)
    return {"E": E_med[order], "width": width[order]}


def plot_levels_window_vs_B_ed(
    *,
    hops: Iterable[Hopping2D],
    fluxes: Iterable[tuple[int, int]],
    mu: float,
    window: float,
    outpath: str,
    nkx: int = 5,
    nky: int = 5,
    B_from_phi: float = 2.0 * np.pi,
    title: str | None = None,
) -> dict:
    """
    Plot all Landau-like levels within an energy window around μ versus B.

    This avoids fragile level-tracking across B by simply scattering every reduced
    level energy E that satisfies |E-μ| <= window at each B.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if window <= 0:
        raise ValueError("window must be > 0.")

    pts_B: list[float] = []
    pts_E: list[float] = []
    pts_w: list[float] = []

    for p, q in fluxes:
        _, _, evals = hofstadter_bands(hops=hops, p=int(p), q=int(q), nkx=nkx, nky=nky)
        reduced = extract_level_energies_from_bands(evals)
        E = np.asarray(reduced["E"], dtype=float)
        w = np.asarray(reduced["width"], dtype=float)

        mask = np.abs(E - float(mu)) <= float(window)
        B = float(B_from_phi) * (float(p) / float(q))
        for Ei, wi in zip(E[mask], w[mask]):
            pts_B.append(B)
            pts_E.append(float(Ei))
            pts_w.append(float(wi))

    B_arr = np.asarray(pts_B, dtype=float)
    E_arr = np.asarray(pts_E, dtype=float)
    w_arr = np.asarray(pts_w, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.4), constrained_layout=True)
    sc = ax.scatter(B_arr, E_arr, c=np.log10(np.maximum(w_arr, 1e-12)), s=18, cmap="viridis")
    ax.axhline(mu, color="k", ls="--", lw=1, alpha=0.7, label="μ")
    ax.set_xlabel("B")
    ax.set_ylabel("E (level energy)")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or f"Levels within |E-μ|≤{window:g} vs B (Peierls+ED)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log10(subband width)")
    ax.legend(frameon=False)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

    return {"B": B_arr, "E": E_arr, "width": w_arr}


def ll_spacing_near_mu_from_levels_window(
    *,
    B: ArrayLike,
    E: ArrayLike,
    mu: float,
    navg: int = 1,
    digits: int = 12,
    dedup_atol: float = 1e-12,
    dedup_rtol: float = 1e-9,
) -> dict:
    """
    Estimate ΔE near μ versus B from the (B,E) scatter produced by
    `plot_levels_window_vs_B_ed`.

    For each B, sort the energies and compute the gaps ΔE_j = E_{j+1}-E_j.
    The "near-μ" gap is chosen by the midpoint criterion: select the gaps whose
    midpoints are closest to μ and average the closest `navg` of them.
    """
    B = np.asarray(B, dtype=float)
    E = np.asarray(E, dtype=float)
    if B.shape != E.shape:
        raise ValueError("B and E must have the same shape.")
    if navg <= 0:
        raise ValueError("navg must be >= 1.")

    if dedup_atol < 0 or dedup_rtol < 0:
        raise ValueError("dedup_atol and dedup_rtol must be >= 0.")

    def dedup_sorted(E_sorted: ArrayLike) -> ArrayLike:
        if E_sorted.size == 0:
            return E_sorted
        out = [float(E_sorted[0])]
        for x in E_sorted[1:]:
            x = float(x)
            y = out[-1]
            tol = float(dedup_atol + dedup_rtol * max(1.0, abs(x), abs(y)))
            if abs(x - y) <= tol:
                continue
            out.append(x)
        return np.asarray(out, dtype=float)

    keys = np.round(B, decimals=digits)
    uniq = np.unique(keys[np.isfinite(keys) & np.isfinite(E)])
    uniq.sort()

    Bout: list[float] = []
    dEout: list[float] = []
    counts: list[int] = []
    unique_counts: list[int] = []

    for k in uniq:
        mask = (keys == k) & np.isfinite(E)
        Ei_all = np.sort(E[mask])
        counts.append(int(Ei_all.size))
        Bout.append(float(np.mean(B[mask])))
        Ei = dedup_sorted(Ei_all)
        unique_counts.append(int(Ei.size))
        if Ei.size < 2:
            dEout.append(float("nan"))
            continue

        gaps = np.diff(Ei)
        mids = 0.5 * (Ei[:-1] + Ei[1:])
        pick = np.argsort(np.abs(mids - float(mu)))[: min(navg, gaps.size)]
        dEout.append(float(np.mean(gaps[pick])))

    return {
        "B": np.asarray(Bout, dtype=float),
        "dE": np.asarray(dEout, dtype=float),
        "count": np.asarray(counts, dtype=int),
        "unique_count": np.asarray(unique_counts, dtype=int),
    }


def plot_ll_spacing_vs_B_from_levels_window(
    *,
    B: ArrayLike,
    E: ArrayLike,
    mu: float,
    outpath: str,
    navg: int = 1,
    dedup_atol: float = 1e-12,
    dedup_rtol: float = 1e-9,
    expected_slope: float | None = None,
    title: str | None = None,
    fit_through_origin: bool = True,
) -> dict:
    """
    Plot ΔE near μ versus B using only the (B,E) scatter from
    `plot_levels_window_vs_B_ed`.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    reduced = ll_spacing_near_mu_from_levels_window(
        B=B,
        E=E,
        mu=mu,
        navg=navg,
        dedup_atol=dedup_atol,
        dedup_rtol=dedup_rtol,
    )
    Bs = reduced["B"]
    dE = reduced["dE"]

    mask = np.isfinite(Bs) & np.isfinite(dE)
    fit_slope = fit_intercept = float("nan")
    if np.count_nonzero(mask) >= 2:
        if fit_through_origin:
            x = Bs[mask]
            y = dE[mask]
            fit_slope = float(np.sum(x * y) / np.sum(x * x))
            fit_intercept = 0.0
        else:
            fit_slope, fit_intercept = np.polyfit(Bs[mask], dE[mask], deg=1)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(Bs, dE, "o-", ms=3, lw=1, label=f"from dots (navg={navg})")
    if expected_slope is not None:
        ax.plot(Bs, expected_slope * Bs, "--", lw=1, label=f"expected slope={expected_slope:.4g}")
    ax.set_xlabel("B")
    ax.set_ylabel("ΔE near μ")
    ax.grid(True, alpha=0.25)
    fit_label = "fit through origin" if fit_through_origin else "fit"
    ax.set_title(title or f"ΔE near μ={mu:g} vs B ({fit_label} slope={fit_slope:.4g})")
    ax.legend(frameon=False)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

    return {
        **reduced,
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
        "fit_through_origin": bool(fit_through_origin),
    }
