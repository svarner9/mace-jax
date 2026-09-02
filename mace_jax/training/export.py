"""Save trained models as bundles and load them back, in the formats mace-jax and mlmd read.

Two on-disk formats, chosen by the model class:

* **MACE / ScaleShiftMACE**: ``<name>.npz`` (flax msgpack of the full module state) next to
  ``<name>.json`` (the architecture). The same files ``mace-jax-from-torch`` writes for converted
  torch models; ``mace_jax.tools.bundle.load_model_bundle`` and ``mlmd.calculators.MACE`` read them.
* **MACELES**: a directory with ``config.json`` (architecture, scale/shift, LES parameters) and
  ``params.msgpack`` (flax msgpack of the ``nnx.Param`` state only); what
  ``mlmd.calculators.MACELES`` reads and what ``mlmd/tools/convert_maceles_torch.py`` writes.

The architecture always travels with the weights. Rebuilding a MACE model from a bare parameter
tree is not supported anywhere in this code base: a wrong ``use_reduced_cg`` or activation constant
loads every parameter by name and shape and gives wrong forces.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization

from mace_jax.nnx_utils import state_to_pure_dict, state_to_serializable_dict
from mace_jax.tools import bundle as bundle_tools
from mace_jax.tools import model_builder
from mace_jax.tools.gin_model import _serialize_config_value


def _read_checkpoint(path: Path) -> tuple[dict, dict]:
    """(model_config, pure parameter/config state) from a trainer ``.ckpt``."""
    with Path(path).open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or 'model_config' not in payload:
        raise ValueError(
            f'{path} is not a mace-jax training checkpoint with a model_config; cannot export a '
            'model without its architecture.'
        )
    state = payload.get('eval_params') or payload.get('params')
    if state is None:
        raise ValueError(f'{path} holds no parameters')
    return dict(payload['model_config']), bundle_tools._unwrap_training_state(state)


def _model_class(config: dict) -> str:
    return config.get('model_class') or config.get('torch_model_class') or 'MACE'


def _full_state(config: dict, state_pure: dict, *, dtype: str | None):
    """Rebuild the module from ``config`` and load ``state_pure`` into its NNX state."""
    if dtype is not None:
        bundle_tools._maybe_set_dtype(dtype)
    config, _, _ = model_builder._normalize_atomic_config(config)
    module = model_builder._build_jax_model(config, rngs=nnx.Rngs(int(config.get('seed', 0))))
    graphdef, state = nnx.split(module)
    bundle_tools._replace_state_with_specials(state, state_pure)
    return module, graphdef, state


def export_bundle(
    source: str | Path | tuple[dict, dict],
    out: str | Path,
    *,
    dtype: str | None = None,
    extra_config: dict | None = None,
) -> Path:
    """Write the bundle for ``source`` (a ``.ckpt`` path, or ``(model_config, params)``) at ``out``.

    Returns the path of the bundle: ``out.npz`` for MACE / ScaleShiftMACE (with ``out.json`` beside
    it), the directory ``out`` for MACELES. ``extra_config`` entries are added to the JSON (e.g.
    provenance).
    """
    if isinstance(source, (str, Path)):
        config, state_pure = _read_checkpoint(Path(source))
        provenance = {'checkpoint': str(Path(source).resolve())}
    else:
        # the trainer hands its parameters as a {params, config} wrapper; the bundle wants one tree
        config, state_pure = dict(source[0]), bundle_tools._unwrap_training_state(source[1])
        provenance = {}
    config = {k: _serialize_config_value(v) for k, v in config.items()}
    if extra_config:
        config.update(extra_config)
    if provenance:
        config.setdefault('source', {}).update(provenance)
    out = Path(out)
    model_class = _model_class(config)
    config['model_class'] = model_class

    _, _, state = _full_state(config, state_pure, dtype=dtype)
    # `state` (not the module) holds the loaded parameters: nnx.replace_by_pure_dict returns new
    # variables inside the State and leaves the module it was split from untouched.

    if model_class == 'LESMACE':
        out.mkdir(parents=True, exist_ok=True)
        params, _ = nnx.split_state(state, nnx.Param, ...)
        (out / 'config.json').write_text(json.dumps(config, indent=1))
        (out / 'params.msgpack').write_bytes(serialization.to_bytes(nnx.to_pure_dict(params)))
        return out

    if out.suffix in ('.npz', '.json'):
        out = out.with_suffix('')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.npz').write_bytes(
        serialization.to_bytes(state_to_serializable_dict(state))
    )
    out.with_suffix('.json').write_text(json.dumps(config, indent=1))
    return out.with_suffix('.npz')


def load_bundle(path: str | Path, dtype: str = 'float32') -> bundle_tools.ModelBundle:
    """Load a bundle written by `export_bundle` (or a ``.ckpt``) as a `ModelBundle`.

    ``bundle.graphdef.apply(bundle.params)(data_dict, compute_force=True)`` evaluates the model;
    `ase_calculator` wraps that for ASE. MACELES directories are loaded through the same path.
    """
    path = Path(path)
    if path.is_dir() and (path / 'params.msgpack').exists():
        config = json.loads((path / 'config.json').read_text())
        bundle_tools._maybe_set_dtype(dtype)
        config, _, _ = model_builder._normalize_atomic_config(config)
        module = model_builder._build_jax_model(config, rngs=nnx.Rngs(int(config.get('seed', 0))))
        graphdef, params, rest = nnx.split(module, nnx.Param, ...)
        template = nnx.to_pure_dict(params)
        restored = serialization.from_bytes(template, (path / 'params.msgpack').read_bytes())
        nnx.replace_by_pure_dict(params, restored)
        state = nnx.merge_state(params, rest)
        return bundle_tools.ModelBundle(
            config=config, params=state_to_pure_dict(state), graphdef=graphdef
        )
    return bundle_tools.load_model_bundle(str(path), dtype)


def ase_calculator(path: str | Path, dtype: str = 'float32'):
    """An ASE calculator for a saved model (energy in eV, forces in eV/A, stress in eV/A^3).

    For ``model='MACE'`` bundles the recorded ``output_mean``/``output_std`` are applied, so every
    model class returns physical units here.
    """
    from mace_jax.calculators.mace import MACEJAXCalculator  # noqa: PLC0415
    from mace_jax.tools import gin_model  # noqa: PLC0415

    bundle = load_bundle(path, dtype)
    config = bundle.config
    num_species = len(config['atomic_numbers'])
    r_max = float(config['r_max'])
    mean = float(config.get('output_mean', 0.0) or 0.0)
    std = float(config.get('output_std', 1.0) or 1.0)
    graphdef = bundle.graphdef

    def model(params, graph, *, compute_force=True, compute_stress=False):
        data = gin_model._graph_to_data(
            graph, num_species=num_species, padded_edge_length=r_max + 1.0
        )
        outputs, _ = graphdef.apply(params)(
            data, compute_force=compute_force, compute_stress=compute_stress
        )
        return gin_model._apply_output_normalization(outputs, graph, mean=mean, std=std)

    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.dtype(dtype)) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x,
        bundle.params,
    )
    graph_transform = None
    if _model_class(config) == 'LESMACE':
        # MACE-LES needs the Ewald k-grid of the structure's cell in the graph; the trainer attaches
        # it per batch, the calculator per structure (its exact size, so a new cell size recompiles).
        from les_jax.ewald import build_kgrid  # noqa: PLC0415

        from .les import attach_kgrid  # noqa: PLC0415

        dl = float((config.get('les') or {}).get('dl', 2.0))

        def graph_transform(graph):
            cells = np.asarray(graph.globals.cell, dtype=np.float64).reshape(-1, 3, 3)
            pbc = np.asarray(graph.globals.pbc, dtype=bool).reshape(-1, 3)
            num_k = max(
                (build_kgrid(c, dl=dl).shape[0] for c, p in zip(cells, pbc) if p.any()),
                default=0,
            )
            if num_k == 0:
                return graph  # non-periodic: real-space branch only
            return attach_kgrid(graph, num_k=num_k, dl=dl)

    return MACEJAXCalculator(
        model=model,
        params=params,
        r_max=r_max,
        default_dtype=dtype,
        atomic_numbers=[int(z) for z in config['atomic_numbers']],
        graph_transform=graph_transform,
    )


def describe_bundle(path: str | Path) -> str:
    """One-paragraph summary of a bundle's architecture (for logs and the export CLI)."""
    path = Path(path)
    if path.is_dir():
        config = json.loads((path / 'config.json').read_text())
    else:
        config = json.loads(path.with_suffix('.json').read_text())
    n_params = None
    try:
        bundle = load_bundle(path, config.get('dtype', 'float32'))
        n_params = int(sum(np.prod(np.shape(x)) for x in jax.tree_util.tree_leaves(bundle.params)))
    except Exception:  # pragma: no cover - description only
        pass
    fields = [
        f"{_model_class(config)}",
        f"r_max={config.get('r_max')}",
        f"Z={config.get('atomic_numbers')}",
        f"{config.get('num_interactions')} layer(s)",
        f"hidden={config.get('hidden_irreps')}",
        f"correlation={config.get('correlation')}",
        f"max_ell={config.get('max_ell')}",
        f"reduced_cg={config.get('use_reduced_cg')}",
    ]
    if 'atomic_inter_scale' in config:
        fields.append(f"scale={config['atomic_inter_scale']}, shift={config['atomic_inter_shift']}")
    if 'les' in config:
        fields.append(f"les={config['les']}")
    if n_params is not None:
        fields.append(f'{n_params} parameters')
    return ', '.join(str(f) for f in fields)


__all__ = ['ase_calculator', 'describe_bundle', 'export_bundle', 'load_bundle']
