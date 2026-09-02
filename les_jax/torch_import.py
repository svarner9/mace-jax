"""torch MACELES → LESMACE (JAX) weight import.

Backbone submodules go through the fork adapter (init_from_torch), the same
path parity_model.py uses for NEMORA. The ONLY new tensors are
``les_readouts.<i>.*`` — byte-mirrors of ``readouts.<i>`` block types, so
they import through the identical adapter path onto the les stack. The
LES/Ewald math has zero parameters.
"""

from __future__ import annotations


def import_lesmace(jax_model, torch_model):
    """Import every weight of a torch MACELES into a fresh LESMACE."""
    from mace_jax.adapters.nnx.torch import init_from_torch

    for attr in ('node_embedding', 'radial_embedding', 'spherical_harmonics',
                 'atomic_energies_fn', 'scale_shift'):
        new_jm, _ = init_from_torch(getattr(jax_model, attr),
                                    getattr(torch_model, attr))
        setattr(jax_model, attr, new_jm)
    for coll in ('interactions', 'products', 'readouts', 'les_readouts'):
        j_list = getattr(jax_model, coll)
        t_list = getattr(torch_model, coll)
        assert len(j_list) == len(t_list), (coll, len(j_list), len(t_list))
        for k in range(len(t_list)):
            new_jm, _ = init_from_torch(j_list[k], t_list[k])
            j_list[k] = new_jm
    return jax_model
