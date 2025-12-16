# Landau levels for tight-binding models (2D)

This folder contains small, self-contained Python code to compute Landau levels for **2D tight-binding models** using **Peierls substitution + exact diagonalization** (Hofstadter / magnetic Bloch approach).

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib` (plots)

If Matplotlib cache warnings appear, set a writable config dir:

```bash
export MPLCONFIGDIR="$PWD/.mplconfig"
```

## Files

- `ll_peierls_ed.py`: Peierls substitution on a square lattice at rational flux `φ=p/q`, magnetic Bloch Hamiltonian, and exact diagonalization helpers
- `demo_peierls_exact_diag.py`: Peierls+ED demo plotting `ΔE(μ)` vs `B`
- `demo_peierls_qwz.py`: multiband (doped QWZ) Peierls+ED demo

## Microscopic method (Peierls + exact diagonalization)

### Idea

1. Specify your model in **real space** as a set of hopping matrices `t(dx,dy)` (multi-orbital supported).
2. Choose a rational flux per plaquette `φ=p/q` (weak field = large `q`).
3. Apply the **Peierls substitution** in Landau gauge and build the **magnetic Bloch Hamiltonian** (Hofstadter form) with a magnetic unit cell of size `q` in `x`.
4. Diagonalize to obtain magnetic subbands; in the weak-field regime these are nearly flat and can be interpreted as Landau levels.

### Quick start (generic multi-band)

```python
import numpy as np
from ll_peierls_ed import Hopping2D, hofstadter_bands, extract_level_energies_from_bands

# Example: 2-orbital model with onsite + nearest-neighbor hoppings (square lattice)
hops = [
    Hopping2D(dx=0, dy=0, mat=np.array([[0.2, 0.0], [0.0, -0.1]], dtype=complex)),
    Hopping2D(dx=1, dy=0, mat=np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=complex)),
    Hopping2D(dx=0, dy=1, mat=np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=complex)),
]

# Flux per plaquette φ = p/q (weak field => large q)
p, q = 1, 41

# Compute magnetic subband energies on a k-grid in the magnetic Brillouin zone
kx_grid, ky_grid, evals = hofstadter_bands(hops=hops, p=p, q=q, nkx=9, nky=9)

# Reduce each subband to a representative "level energy" (median over k)
reduced = extract_level_energies_from_bands(evals)
level_E = reduced["E"]
level_width = reduced["width"]  # residual dispersion of each subband

mu = 0.0
idx = np.argmin(np.abs(level_E - mu))
print("closest level:", level_E[idx], "bandwidth:", level_width[idx])
```

### Units / conventions

`ll_peierls_ed.py` uses the Hofstadter convention `φ=p/q` (flux per plaquette in units of the flux quantum `Φ0`), with Peierls phases `exp(i 2π φ …)`.

If you work in units `e=ħ=a=1`, a common convention is that the phase around one plaquette is `exp(i B)`, so:

`B = 2π φ`.

### Run the Peierls+ED demo

```bash
python demo_peierls_exact_diag.py
```

This writes `ll_spacing_vs_B_peierls_tb.png` and prints the fitted slope of `ΔE` vs `B` in the weak-field regime.

### Multiband example: doped QWZ model

`demo_peierls_qwz.py` encodes the 2-band Qi–Wu–Zhang (QWZ) model as real-space hoppings and plots several Landau-like levels near a doped chemical potential versus `B`:

```bash
python demo_peierls_qwz.py
```

### Limitations

- Currently targets a **square lattice** and uses a **Landau-gauge magnetic unit cell of size `q` in x**.
- Dense diagonalization scales like `O((q*norb)^3 * Nk)`; very large `q` will be slow without sparse methods.
- If magnetic subbands are not flat (large `width` near `μ`), you are not yet in the “LL-like” weak-field regime for your chosen `q`.
