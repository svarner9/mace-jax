"""JAX port of the LES Ewald long-range energy (les/module/ewald.py).

Parity-exact reimplementation of torch `les.module.ewald.Ewald` with the
config every trained MACE-LES uses (no script overrides `--les_arguments`):

    sigma=1.0, dl=2.0, remove_self_interaction=True, norm_factor=90.4756
    k_sq_max = (2*pi/dl)**2

Zero trainable parameters — pure math. Two branches, selected per graph on
``det(cell) < 1e-6`` (torch condition verbatim; negative det → real-space):

* periodic (``potential_triclinic``): reciprocal-space Gaussian-smeared
  Coulomb over 0 < k² ≤ k_sq_max, G=0 dropped (tinfoil), self-energy
  subtracted. The torch code sums a k-hemisphere with factor 2; here the
  FULL k-grid is summed with factor 1 — bit-equivalent (|S(-k)|²=|S(k)|²,
  kfac even) and it kills the argmax hemisphere logic.
* non-periodic (``potential_realspace``): O(N²) all-pairs erf-screened
  Coulomb; the self term is already absent, matching the "self removed"
  convention of the reciprocal branch.

NO 2π factor in the reciprocal assembly (the torch code comment claims one;
the code has none) and norm_factor is exactly 90.4756 = 1/(2ε₀).

The reference path (`ewald_energy_ref`) is host-driven (per-graph python
loop, host-built k-grid) for fp64 parity oracles. The static-shape jitted
path for production lives in `les_jax/ewald_static.py` (Stage S3).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

# k.r reaches tens of radians, so the structure factor needs full float32 products: on NVIDIA GPUs JAX
# otherwise runs float32 matmuls in TF32 (10-bit mantissa), which put ~0.1 eV/A errors into the LES
# forces (mlmd, 2026-08-27). These products are tiny (N x K x 3), so HIGHEST costs nothing.
_EXACT = jax.lax.Precision.HIGHEST

DEFAULT_SIGMA = 1.0
DEFAULT_DL = 2.0
DEFAULT_NORM_FACTOR = 90.4756
TWO_PI = 2.0 * math.pi


def potential_realspace(
    r: jnp.ndarray,  # [n_atoms, 3]
    q: jnp.ndarray,  # [n_atoms, n_q]
    *,
    sigma: float = DEFAULT_SIGMA,
    remove_self_interaction: bool = True,
    norm_factor: float = DEFAULT_NORM_FACTOR,
) -> jnp.ndarray:
    """Non-periodic all-pairs erf-screened Coulomb (ewald.py:66-98). Scalar out."""
    n = r.shape[0]
    eye = jnp.eye(n, dtype=r.dtype)
    # torch adds 1e-6 to every component of the diagonal r_ii (safe Hessian);
    # the diagonal is masked out of the energy, epsilon only guards the grad.
    r_ij = r[None, :, :] - r[:, None, :] + eye[:, :, None] * 1.0e-6
    dist = jnp.sqrt(jnp.sum(r_ij * r_ij, axis=-1))
    conv = erf(dist / (sigma * math.sqrt(2.0)))
    pair = (dist ** -1.0) * conv  # [n, n]
    qq = q[None, :, :] * q[:, None, :]  # [n, n, n_q]
    off_diag = 1.0 - eye
    pot = jnp.sum(qq * (pair * off_diag)[:, :, None]) / TWO_PI / 2.0
    if not remove_self_interaction:
        pot = pot + jnp.sum(q * q) / (sigma * TWO_PI ** 1.5)
    return pot * norm_factor


def build_kgrid(cell: np.ndarray, *, dl: float = DEFAULT_DL) -> np.ndarray:
    """Host-side integer frequency grid [-Nk,Nk]³ (ewald.py:109-116).

    Nk[a] = max(1, floor(||cell_row_a|| / dl)); torch `int()` truncates, and
    norms are positive, so floor matches. Returns int64 [M, 3] (unfiltered —
    the 0<k²≤k_sq_max mask depends on G and is applied in the energy fn).
    """
    cell = np.asarray(cell, dtype=np.float64)
    norms = np.linalg.norm(cell, axis=1)
    nk = [max(1, int(math.floor(float(v) / dl))) for v in norms]
    axes = [np.arange(-k, k + 1, dtype=np.int64) for k in nk]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return grid


def potential_triclinic(
    r: jnp.ndarray,  # [n_atoms, 3]
    q: jnp.ndarray,  # [n_atoms, n_q]
    cell: jnp.ndarray,  # [3, 3]
    nvec: jnp.ndarray,  # [M, 3] integer frequency grid (build_kgrid(cell))
    *,
    sigma: float = DEFAULT_SIGMA,
    dl: float = DEFAULT_DL,
    remove_self_interaction: bool = True,
    norm_factor: float = DEFAULT_NORM_FACTOR,
) -> jnp.ndarray:
    """Periodic reciprocal-space branch (ewald.py:101-159). Scalar out.

    Full-space k-sum with factor 1 (bit-equivalent to torch's hemisphere ×2).
    Rows of ``nvec`` outside 0 < k² ≤ k_sq_max contribute exactly 0 via the
    mask, so a padded grid (extra zero rows) is safe.
    """
    k_sq_max = (TWO_PI / dl) ** 2
    g_mat = TWO_PI * jnp.linalg.inv(cell).T
    kvec = jnp.matmul(nvec.astype(r.dtype), g_mat, precision=_EXACT)  # [M, 3]
    k_sq = jnp.sum(kvec * kvec, axis=1)  # [M]
    mask = (k_sq > 0.0) & (k_sq <= k_sq_max)

    k_dot_r = jnp.matmul(r, kvec.T, precision=_EXACT)  # [n, M]
    cos_kr = jnp.cos(k_dot_r)
    sin_kr = jnp.sin(k_dot_r)
    s_real = jnp.sum(q[:, :, None] * cos_kr[:, None, :], axis=0)  # [n_q, M]
    s_imag = jnp.sum(q[:, :, None] * sin_kr[:, None, :], axis=0)
    s_sq = s_real ** 2 + s_imag ** 2

    safe_k_sq = jnp.where(mask, k_sq, 1.0)
    kfac = jnp.where(mask, jnp.exp(-(sigma ** 2 / 2.0) * safe_k_sq) / safe_k_sq, 0.0)

    volume = jnp.linalg.det(cell)
    pot = jnp.sum(kfac[None, :] * s_sq, axis=1) / volume  # [n_q]
    if remove_self_interaction:
        pot = pot - jnp.sum(q * q) / (sigma * TWO_PI ** 1.5)
    return jnp.sum(pot) * norm_factor


def ewald_energy_batched(
    q: jnp.ndarray,  # [n_atoms] or [n_atoms, n_q]
    positions: jnp.ndarray,  # [n_atoms, 3]
    cell: jnp.ndarray,  # [n_graphs, 3, 3] — LES cell (rows zeroed for non-PBC)
    batch: jnp.ndarray,  # [n_atoms] int graph index
    nvec: jnp.ndarray,  # [n_graphs, num_k, 3] integer grid, zero-padded
    num_graphs: int,
    *,
    node_mask: jnp.ndarray | None = None,  # [n_atoms] 1/0 for padded atoms
    include_reciprocal: bool = True,  # static: batch may contain PBC graphs
    include_realspace: bool = True,  # static: batch may contain non-PBC graphs
    sigma: float = DEFAULT_SIGMA,
    dl: float = DEFAULT_DL,
    remove_self_interaction: bool = True,
    norm_factor: float = DEFAULT_NORM_FACTOR,
) -> jnp.ndarray:
    """Jittable static-shape LES energy, per graph → [n_graphs].

    Branch per graph on det(cell) < 1e-6 (torch condition): reciprocal for
    periodic graphs, all-pairs erf real-space for the rest. ``nvec`` rows
    beyond a graph's true k-grid must be zero (k²=0 → masked out); the
    ``include_*`` flags are STATIC and prune dead branches at trace time
    (host collate knows dataset periodicity).
    """
    from jax.ops import segment_sum

    if q.ndim == 1:
        q = q[:, None]
    dt = positions.dtype
    q = q.astype(dt)
    if node_mask is None:
        qm = q
        amask = jnp.ones((positions.shape[0],), dtype=dt)
    else:
        amask = jnp.asarray(node_mask, dtype=dt)
        qm = q * amask[:, None]

    det = jnp.linalg.det(cell)
    is_pbc = det >= 1e-6  # [n_graphs]
    energy = jnp.zeros((num_graphs,), dtype=dt)
    # per-graph Σq² (self-energy; reciprocal branch subtracts it)
    q_sq_graph = segment_sum(jnp.sum(qm * qm, axis=1), batch,
                             num_segments=num_graphs)

    if include_reciprocal:
        k_sq_max = (TWO_PI / dl) ** 2
        eye = jnp.eye(3, dtype=dt)
        safe_cell = jnp.where(is_pbc[:, None, None], cell, eye)
        g_mat = TWO_PI * jnp.transpose(jnp.linalg.inv(safe_cell), (0, 2, 1))
        kvec = jnp.einsum('gkc,gcd->gkd', nvec.astype(dt), g_mat, precision=_EXACT)
        k_sq = jnp.sum(kvec * kvec, axis=-1)  # [g, K]
        kmask = (k_sq > 0.0) & (k_sq <= k_sq_max) & is_pbc[:, None]
        safe_k_sq = jnp.where(kmask, k_sq, 1.0)
        kfac = jnp.where(kmask,
                         jnp.exp(-(sigma ** 2 / 2.0) * safe_k_sq) / safe_k_sq,
                         0.0)  # [g, K]
        k_dot_r = jnp.einsum('nd,nkd->nk', positions, kvec[batch], precision=_EXACT)  # [n, K]
        s_real = segment_sum(qm[:, :, None] * jnp.cos(k_dot_r)[:, None, :],
                             batch, num_segments=num_graphs)  # [g, nq, K]
        s_imag = segment_sum(qm[:, :, None] * jnp.sin(k_dot_r)[:, None, :],
                             batch, num_segments=num_graphs)
        s_sq = s_real ** 2 + s_imag ** 2
        volume = jnp.where(is_pbc, det, 1.0)
        pot_recip = jnp.sum(kfac[:, None, :] * s_sq, axis=(1, 2)) / volume
        if remove_self_interaction:
            pot_recip = pot_recip - q_sq_graph / (sigma * TWO_PI ** 1.5)
        energy = energy + jnp.where(is_pbc, pot_recip, 0.0)

    if include_realspace:
        n = positions.shape[0]
        eye_n = jnp.eye(n, dtype=dt)
        same_graph = (batch[:, None] == batch[None, :]).astype(dt)
        pmask = same_graph * (1.0 - eye_n) * amask[:, None] * amask[None, :]
        r_ij = (positions[None, :, :] - positions[:, None, :]
                + eye_n[:, :, None] * 1.0e-6)
        # Padded atoms share identical positions: their pair distance is 0 and
        # erf(0)/0 → NaN, which survives multiplication by the 0 mask (and its
        # sqrt gradient is NaN regardless). Guard r² on masked-out pairs
        # BEFORE sqrt — real-pair values and gradients are untouched.
        r_sq = jnp.sum(r_ij * r_ij, axis=-1)
        dist = jnp.sqrt(jnp.where(pmask > 0, r_sq, 1.0))
        conv = erf(dist / (sigma * math.sqrt(2.0)))
        pair = conv / dist  # [n, n]
        qq = jnp.sum(q[None, :, :] * q[:, None, :], axis=-1)  # Σ_nq q_i q_j
        row_e = jnp.sum(qq * pair * pmask, axis=1)  # [n]
        pot_real = segment_sum(row_e, batch,
                               num_segments=num_graphs) / TWO_PI / 2.0
        if not remove_self_interaction:
            pot_real = pot_real + q_sq_graph / (sigma * TWO_PI ** 1.5)
        energy = energy + jnp.where(is_pbc, 0.0, pot_real)

    return energy * norm_factor


def ewald_energy_ref(
    q,  # [n_atoms] or [n_atoms, n_q]
    r,  # [n_atoms, 3]
    cell,  # [n_graphs, 3, 3] or None
    batch=None,  # [n_atoms] int, host-readable (np array / concrete)
    *,
    sigma: float = DEFAULT_SIGMA,
    dl: float = DEFAULT_DL,
    remove_self_interaction: bool = True,
    norm_factor: float = DEFAULT_NORM_FACTOR,
) -> jnp.ndarray:
    """Reference multi-graph driver mirroring torch Ewald.forward (ewald.py:28-64).

    Host python loop over graphs, host branch on det(cell) — NOT jittable.
    Returns [n_graphs] energies. For parity oracles only.
    """
    q = jnp.asarray(q)
    if q.ndim == 1:
        q = q[:, None]
    r = jnp.asarray(r)
    n = r.shape[0]
    if batch is None:
        batch_np = np.zeros(n, dtype=np.int64)
    else:
        batch_np = np.asarray(batch)
    kw = dict(sigma=sigma, remove_self_interaction=remove_self_interaction,
              norm_factor=norm_factor)
    energies = []
    for gi in np.unique(batch_np):
        sel = np.nonzero(batch_np == gi)[0]
        r_g, q_g = r[sel], q[sel]
        if cell is None:
            energies.append(potential_realspace(r_g, q_g, **kw))
            continue
        cell_g = jnp.asarray(cell)[int(gi)]
        if float(np.linalg.det(np.asarray(cell_g, dtype=np.float64))) < 1e-6:
            energies.append(potential_realspace(r_g, q_g, **kw))
        else:
            nvec = jnp.asarray(build_kgrid(np.asarray(cell_g), dl=dl))
            energies.append(
                potential_triclinic(r_g, q_g, cell_g, nvec, dl=dl, **kw))
    return jnp.stack(energies)
