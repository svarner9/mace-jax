"""les_jax: JAX port of the LES (Latent Ewald Summation) long-range energy.

Parity target: torch `les.module.ewald.Ewald` (defaults sigma=1.0, dl=2.0,
remove_self_interaction=True, norm_factor=90.4756) as used by torch MACELES.
"""

from .ewald import (
    DEFAULT_DL,
    DEFAULT_NORM_FACTOR,
    DEFAULT_SIGMA,
    build_kgrid,
    ewald_energy_batched,
    ewald_energy_ref,
    potential_realspace,
    potential_triclinic,
)

__all__ = [
    'DEFAULT_DL',
    'DEFAULT_NORM_FACTOR',
    'DEFAULT_SIGMA',
    'build_kgrid',
    'ewald_energy_batched',
    'ewald_energy_ref',
    'potential_realspace',
    'potential_triclinic',
]
