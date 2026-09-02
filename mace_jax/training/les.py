"""MACE-LES batching: attach the reciprocal-space k-grid to every batch.

`les_jax.model.LESMACE` evaluates the Ewald sum on an integer frequency grid ``n`` with
``k = n G`` (``G`` the reciprocal cell), truncated at ``|k| <= 2 pi / dl``. The grid extent depends
on the cell, and a compiled JAX program needs a static shape, so:

* `scan_hdf5_cells` reads every cell in the HDF5 shards once and returns the largest grid
  (``num_k``) any structure needs, and whether the dataset holds periodic / non-periodic frames;
* `attach_kgrid` builds each graph's exact grid (`les_jax.ewald.build_kgrid`, identical to torch
  ``les``) on the host and zero-pads it to ``num_k``. Padded rows have ``k = 0`` and are masked out
  in the energy, so the sum is unchanged;
* `LESLoader` wraps a streaming loader and does this to every batch it yields.

One dataset-wide ``num_k`` means one compiled program. A dataset mixing very different cell sizes
pays the largest grid on every batch; that is a speed cost only, never a correctness one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class LESDatasetInfo:
    num_k: int
    has_pbc: bool
    has_nonpbc: bool
    n_structures: int


def _iter_hdf5_cells(paths: Iterable[Path]):
    for path in paths:
        with h5py.File(path, 'r') as handle:
            for batch_name in handle:
                batch = handle[batch_name]
                if not isinstance(batch, h5py.Group):
                    continue
                for config_name in batch:
                    group = batch[config_name]
                    if 'cell' not in group:
                        continue
                    cell = group['cell'][()]
                    pbc = group['pbc'][()] if 'pbc' in group else None
                    yield np.asarray(cell, dtype=np.float64).reshape(3, 3), (
                        None if pbc is None else np.asarray(pbc).astype(bool).reshape(-1)[:3]
                    )


def scan_hdf5_cells(paths: Iterable[Path], *, dl: float) -> LESDatasetInfo:
    """Largest k-grid any structure in ``paths`` needs, and the dataset's periodicity."""
    from les_jax.ewald import build_kgrid  # noqa: PLC0415

    num_k = 0
    has_pbc = False
    has_nonpbc = False
    count = 0
    for cell, pbc in _iter_hdf5_cells(paths):
        count += 1
        periodic = bool(np.any(pbc)) if pbc is not None else bool(abs(np.linalg.det(cell)) > 1e-6)
        if periodic:
            has_pbc = True
            num_k = max(num_k, int(build_kgrid(cell, dl=dl).shape[0]))
        else:
            has_nonpbc = True
    return LESDatasetInfo(num_k=num_k, has_pbc=has_pbc, has_nonpbc=has_nonpbc, n_structures=count)


def attach_kgrid(graph, *, num_k: int, dl: float):
    """Return ``graph`` with ``globals.les_nvec`` set ([n_graphs, num_k, 3] int32, zero padded)."""
    from les_jax.ewald import build_kgrid  # noqa: PLC0415

    cells = np.asarray(graph.globals.cell, dtype=np.float64).reshape(-1, 3, 3)
    pbc = getattr(graph.globals, 'pbc', None)
    pbc = (
        np.asarray(pbc, dtype=bool).reshape(-1, 3)
        if pbc is not None
        else np.ones((cells.shape[0], 3), dtype=bool)
    )
    nvec = np.zeros((cells.shape[0], num_k, 3), dtype=np.int32)
    for i, cell in enumerate(cells):
        if not pbc[i].any() or abs(np.linalg.det(cell)) < 1e-6:
            continue  # padding graph or non-periodic structure: no reciprocal-space term
        grid = build_kgrid(cell, dl=dl)
        if grid.shape[0] > num_k:
            raise ValueError(
                f'structure needs a {grid.shape[0]}-point k-grid but the dataset scan found at most '
                f'{num_k}; the cell is not in the training/validation/test data that was scanned.'
            )
        nvec[i, : grid.shape[0]] = grid
    return graph._replace(globals=graph.globals._replace(les_nvec=nvec))


class LESLoader:
    """A streaming loader whose batches carry the MACE-LES k-grid. Everything else is delegated."""

    def __init__(self, loader, *, num_k: int, dl: float):
        self._loader = loader
        self._num_k = int(num_k)
        self._dl = float(dl)

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader)

    def _wrap(self, iterator):
        for batch in iterator:
            yield attach_kgrid(batch, num_k=self._num_k, dl=self._dl)

    def iter_batches(self, *args, **kwargs):
        inner = self._loader.iter_batches(*args, **kwargs)
        wrapped = self._wrap(inner)
        hint = getattr(inner, 'total_batches_hint', None)
        if hint is None:
            return wrapped
        from mace_jax.data.streaming_loader import BatchIteratorWrapper  # noqa: PLC0415

        return BatchIteratorWrapper(wrapped, int(hint))

    def __iter__(self):
        return self._wrap(iter(self._loader))

    def shard(self, *args, **kwargs):
        return LESLoader(self._loader.shard(*args, **kwargs), num_k=self._num_k, dl=self._dl)

    def split_by_heads(self):
        return {
            name: LESLoader(loader, num_k=self._num_k, dl=self._dl)
            for name, loader in self._loader.split_by_heads().items()
        }


def wrap_loaders(loaders, hdf5_paths: Iterable[Path], *, dl: float):
    """Wrap ``loaders`` (train, valid, test; any may be None) for MACE-LES training.

    Returns ``(wrapped_loaders, info)``. If no structure in ``hdf5_paths`` is periodic, the loaders
    are returned unchanged: LESMACE then evaluates the real-space branch only.
    """
    info = scan_hdf5_cells(hdf5_paths, dl=dl)
    logging.info(
        'MACE-LES k-grid: %d structures scanned, num_k=%d, periodic=%s, non-periodic=%s',
        info.n_structures,
        info.num_k,
        info.has_pbc,
        info.has_nonpbc,
    )
    if not info.has_pbc:
        return tuple(loaders), info
    wrapped = tuple(
        None if loader is None else LESLoader(loader, num_k=info.num_k, dl=dl)
        for loader in loaders
    )
    return wrapped, info


__all__ = ['LESDatasetInfo', 'LESLoader', 'attach_kgrid', 'scan_hdf5_cells', 'wrap_loaders']
