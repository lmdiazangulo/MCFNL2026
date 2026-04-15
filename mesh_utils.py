"""
mesh_utils.py – Helpers for building 1-D non-uniform FDTD meshes
=================================================================
Every function returns a 1-D numpy array of node positions suitable
for passing to FDTD1DNonUniform(x, ...).

Available generators
--------------------
uniform_mesh      – baseline uniform spacing
cosine_mesh       – smooth refinement at one or both ends (recommended)
geometric_mesh    – geometric progression left→right
step_mesh         – abrupt size jump at a split point  (worst case test)
custom_mesh       – arbitrary spacing via a user-supplied callable
"""

import numpy as np


# ======================================================================
# 1. Uniform (reference)
# ======================================================================
def uniform_mesh(x_start: float, x_end: float, N: int) -> np.ndarray:
    """Standard uniform mesh – N nodes from x_start to x_end."""
    return np.linspace(x_start, x_end, N)


# ======================================================================
# 2. Cosine (smooth, recommended for production non-uniform tests)
# ======================================================================
def cosine_mesh(x_start: float, x_end: float, N: int,
                refinement: str = 'both') -> np.ndarray:
    """
    Cosine-spaced mesh.  Produces smooth transitions – the ratio between
    neighbouring cells never exceeds ~pi/2, so numerical reflections are
    very small.

    Parameters
    ----------
    refinement : {'both', 'left', 'right'}
        Where the mesh is densest.
    """
    if refinement == 'both':
        t = np.linspace(0.0, np.pi, N)
        xi = (1.0 - np.cos(t)) / 2.0
    elif refinement == 'left':
        t  = np.linspace(0.0, np.pi / 2.0, N)
        xi = 1.0 - np.cos(t)
    elif refinement == 'right':
        t  = np.linspace(0.0, np.pi / 2.0, N)
        xi = np.sin(t)
    else:
        raise ValueError(f"refinement must be 'both', 'left' or 'right', got {refinement!r}")
    return x_start + (x_end - x_start) * xi


# ======================================================================
# 3. Geometric (moderate gradation)
# ======================================================================
def geometric_mesh(x_start: float, x_end: float, N: int,
                   ratio: float = 1.05) -> np.ndarray:
    """
    Geometric mesh: each cell is `ratio` times the previous one.
    ratio > 1  →  finer at the left, coarser at the right.
    ratio < 1  →  coarser at the left, finer at the right.
    ratio = 1  →  falls back to uniform.

    Parameters
    ----------
    ratio : float
        Size ratio between consecutive cells (r = dx_{i+1}/dx_i).
    """
    if abs(ratio - 1.0) < 1e-10:
        return uniform_mesh(x_start, x_end, N)

    n_cells = N - 1
    L = x_end - x_start
    # dx0 * (r^n - 1)/(r - 1) = L
    dx0 = L * (ratio - 1.0) / (ratio ** n_cells - 1.0)
    spacings = dx0 * ratio ** np.arange(n_cells)
    return np.concatenate([[x_start], x_start + np.cumsum(spacings)])


# ======================================================================
# 4. Step mesh (abrupt jump – worst case for numerical reflections)
# ======================================================================
def step_mesh(x_start: float, x_end: float, N: int,
              split: float = 0.5, ratio: float = 3.0) -> np.ndarray:
    """
    Two-zone mesh with an abrupt cell-size change at `split` (fraction
    of the domain).  This intentionally violates the smooth-variation
    assumption and is used to study when non-uniform meshes degrade.

    Parameters
    ----------
    split : float
        Position of the jump as a fraction in (0, 1).
    ratio : float
        dx_right / dx_left.  Values > 2 produce visible reflections.
    """
    x_split = x_start + split * (x_end - x_start)
    L1 = x_split - x_start
    L2 = x_end - x_split

    # Solve for N1 (nodes in left zone) so that total nodes = N.
    #   (N1-1)*dx1 = L1  and  (N-N1)*dx2 = L2  with dx2 = ratio*dx1
    # => N1 = round( (ratio*N*L1 + L2) / (L2 + ratio*L1) ) + 1
    N1 = int(round((ratio * N * L1 + L2) / (L2 + ratio * L1))) + 1
    N1 = max(2, min(N - 2, N1))
    N2 = N - N1 + 1          # +1 because the split node is shared

    x_left  = np.linspace(x_start, x_split, N1)
    x_right = np.linspace(x_split, x_end,   N2)
    return np.concatenate([x_left, x_right[1:]])


# ======================================================================
# 5. Custom mesh via spacing function
# ======================================================================
def custom_mesh(x_start: float, x_end: float, N: int,
                spacing_func) -> np.ndarray:
    """
    Build a mesh from an arbitrary spacing function.

    Parameters
    ----------
    spacing_func : callable
        ``spacing_func(xi)`` where ``xi`` ∈ [0, 1] (normalised position).
        Returns a *relative local cell size* – larger value → coarser mesh
        at that location.

    Examples
    --------
    # Denser in the centre
    x = custom_mesh(0, 10, 200,
                    lambda xi: 1 - 0.8*np.exp(-0.5*((xi-0.5)/0.1)**2))

    # Denser near x = 0.25 (e.g. around a material interface)
    x = custom_mesh(0, 10, 200,
                    lambda xi: 0.2 + np.abs(xi - 0.25))
    """
    xi_fine = np.linspace(0.0, 1.0, 50_000)
    s = np.asarray(spacing_func(xi_fine), dtype=float)
    if np.any(s <= 0):
        raise ValueError("spacing_func must be strictly positive everywhere.")
    # Integrate (∝ arc length in "spacing space")
    cumulative = np.concatenate([[0.0],
                                 np.cumsum(0.5 * np.diff(xi_fine) * (s[:-1] + s[1:]))])
    cumulative /= cumulative[-1]   # normalise to [0, 1]

    t_target = np.linspace(0.0, 1.0, N)
    xi_nodes = np.interp(t_target, cumulative, xi_fine)
    return x_start + xi_nodes * (x_end - x_start)


# ======================================================================
# Diagnostics
# ======================================================================
def mesh_stats(x: np.ndarray, label: str = '') -> dict:
    """
    Print and return statistics about mesh spacing.

    Returns a dict with keys: N, dx_min, dx_max, dx_mean, max_ratio.
    """
    dx = np.diff(x)
    stats = {
        'N':         len(x),
        'dx_min':    dx.min(),
        'dx_max':    dx.max(),
        'dx_mean':   dx.mean(),
        'max_ratio': dx.max() / dx.min(),
    }
    tag = f"[{label}] " if label else ""
    print(f"{tag}Mesh stats:")
    print(f"  N nodes   : {stats['N']}")
    print(f"  dx min    : {stats['dx_min']:.6f}")
    print(f"  dx max    : {stats['dx_max']:.6f}")
    print(f"  dx mean   : {stats['dx_mean']:.6f}")
    print(f"  max/min   : {stats['max_ratio']:.3f}")
    return stats