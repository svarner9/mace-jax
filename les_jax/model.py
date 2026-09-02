"""LESMACE (JAX): mace-jax ScaleShiftMACE + latent-charge head + LES Ewald.

Mirrors torch MACELES (mace/modules/extensions.py:77-299):

* ``les_readouts`` — a second readout stack mirroring the energy ``readouts``
  one-for-one (same block classes/irreps, independent weights). Per-atom
  latent charge q_i = Σ_layers les_readout_l(node_feats_l)  → [n_atoms].
* LES Ewald energy E_LR(q, positions, cell) — parameter-free (les_jax.ewald).
* Energy assembly (extensions.py:236-257 — DIFFERS from NEMORA):

      inter_e = scatter( scale_shift( Σ readouts + pair ) )
      total   = e0 + inter_e + E_LR          # E_LR POST scale_shift, UNSCALED

Non-PBC routing mirrors torch (extensions.py:136-143): cell rows are zeroed
for graphs whose ``pbc`` is all-False, and the Ewald branch is selected per
graph on det(cell) < 1e-6.

k-grid protocol (static-shape jit): the caller puts the padded integer
frequency grid in ``data['les_nvec']`` ([n_graphs, num_k, 3], zero rows are
inert; build with `les_jax.ewald.build_kgrid` + pad). No ``les_nvec`` →
real-space branch only (non-PBC batch). ``data['les_has_nonpbc']`` (python
bool, static) forces the real-space branch on for mixed batches; default =
(les_nvec is None).

Forces via @add_output_interface: E_LR reads ``data['positions']`` (the array
the force wrapper differentiates), so LES forces are free. Stress through the
wrapper does NOT capture the LES cell derivative — do not trust LES stress
(torch-line training here uses energy+forces only).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import nnx

from mace_jax.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace_jax.modules.models import ScaleShiftMACE
from mace_jax.modules.utils import add_output_interface
from mace_jax.tools.scatter import scatter_sum

from .ewald import (
    DEFAULT_DL,
    DEFAULT_NORM_FACTOR,
    DEFAULT_SIGMA,
    ewald_energy_batched,
)


def _mirror_readout(readout, *, rngs: nnx.Rngs):
    """Fresh readout block of the same type/irreps (torch _copy_mace_readout)."""
    if isinstance(readout, NonLinearReadoutBlock):
        return NonLinearReadoutBlock(
            readout.irreps_in,
            readout.MLP_irreps,
            readout.gate,
            readout.irrep_out,
            readout.num_heads,
            readout.cueq_config,
            rngs=rngs,
        )
    if isinstance(readout, LinearReadoutBlock):
        return LinearReadoutBlock(
            readout.irreps_in,
            readout.irrep_out,
            readout.cueq_config,
            rngs=rngs,
        )
    raise TypeError(f'Unsupported readout type: {type(readout)}')


@add_output_interface
class LESMACE(ScaleShiftMACE):
    """ScaleShiftMACE + LES latent-charge head (torch MACELES mirror)."""

    def __init__(
        self,
        *,
        les_sigma: float = DEFAULT_SIGMA,
        les_dl: float = DEFAULT_DL,
        les_remove_self_interaction: bool = True,
        les_norm_factor: float = DEFAULT_NORM_FACTOR,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        super().__init__(rngs=rngs, **kwargs)
        self.les_sigma = float(les_sigma)
        self.les_dl = float(les_dl)
        self.les_remove_self_interaction = bool(les_remove_self_interaction)
        self.les_norm_factor = float(les_norm_factor)
        self.les_readouts = nnx.List(
            [_mirror_readout(r, rngs=rngs) for r in self.readouts]
        )

    def __call__(
        self,
        data: dict[str, Any],
        *,
        lammps_mliap: bool = False,
        lammps_class: Any | None = None,
        compute_node_feats: bool = True,
    ) -> dict[str, jnp.ndarray | None]:
        core = self._forward_core(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=lammps_class,
            compute_pair_energy=False,
        )
        ctx = core['ctx']
        node_heads = core['node_heads']
        num_atoms_arange = core['num_atoms_arange']
        node_e0 = core['node_e0']
        e0 = core['e0']
        node_feats = core['node_feats']
        node_feats_list = core['node_feats_list']

        node_energies_list = [core['pair_node_energy']]
        node_qs_list = []
        for idx, (readout, les_readout) in enumerate(
            zip(self.readouts, self.les_readouts)
        ):
            feat_idx = -1 if len(self.readouts) == 1 else idx
            feats = node_feats_list[feat_idx]
            node_energies_list.append(
                readout(feats, node_heads)[num_atoms_arange, node_heads]
            )
            node_qs_list.append(
                les_readout(feats, node_heads)[num_atoms_arange, node_heads]
            )

        node_feats_out = None
        if compute_node_feats:
            node_feats_out = (
                jnp.concatenate(node_feats_list, axis=-1)
                if node_feats_list
                else node_feats
            )
        node_inter_es = jnp.sum(jnp.stack(node_energies_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(
            node_inter_es,
            index=data['batch'],
            dim=-1,
            dim_size=ctx.num_graphs,
            indices_are_sorted=True,
        )

        # --- LES: latent charges + Ewald energy (post-scale_shift, unscaled) ---
        les_q = jnp.sum(jnp.stack(node_qs_list, axis=1), axis=1)  # [n_atoms]
        positions = jnp.asarray(data['positions'])
        cell_les = jnp.asarray(data['cell'], dtype=positions.dtype).reshape(
            -1, 3, 3
        )
        pbc = data.get('pbc')
        if pbc is not None:
            no_pbc = ~jnp.any(
                jnp.asarray(pbc, dtype=bool).reshape(-1, 3), axis=-1
            )
            cell_les = jnp.where(no_pbc[:, None, None],
                                 jnp.zeros_like(cell_les), cell_les)

        nvec = data.get('les_nvec')
        include_reciprocal = nvec is not None
        include_realspace = bool(data.get('les_has_nonpbc',
                                          not include_reciprocal))
        if nvec is None:
            nvec = jnp.zeros((ctx.num_graphs, 1, 3), dtype=positions.dtype)
        les_energy = ewald_energy_batched(
            les_q,
            positions,
            cell_les,
            jnp.asarray(data['batch']),
            jnp.asarray(nvec),
            ctx.num_graphs,
            node_mask=data.get('node_mask'),
            include_reciprocal=include_reciprocal,
            include_realspace=include_realspace,
            sigma=self.les_sigma,
            dl=self.les_dl,
            remove_self_interaction=self.les_remove_self_interaction,
            norm_factor=self.les_norm_factor,
        )

        total_energy = e0 + inter_e + les_energy
        node_energy = node_e0 + node_inter_es
        contributions = jnp.stack((e0, inter_e), axis=-1)
        return {
            'energy': total_energy,
            'node_energy': node_energy,
            'contributions': contributions,
            'node_feats': node_feats_out,
            'interaction_energy': inter_e,
            'les_energy': les_energy,
            'latent_charges': les_q,
            'displacement': ctx.displacement,
            'lammps_natoms': ctx.interaction_kwargs.lammps_natoms,
        }
