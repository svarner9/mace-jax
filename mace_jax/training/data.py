"""Turn the user's data files into what the streaming trainer reads.

The trainer streams fixed-shape batches from HDF5 shards and needs a ``statistics.json`` (E0s,
average neighbor count, energy shift/scale) computed over the *whole* training set. torch MACE
users hand over XYZ files, so `prepare_data` does the conversion for them, once: the shards and
statistics land in ``<results_dir>/data/<key>/`` where ``<key>`` hashes the input files (path, size,
mtime) and every preprocessing option, so a rerun with the same inputs skips the step and a change
in ``r_max`` or the property keys invalidates it.

An HDF5 directory produced by ``mace-jax-preprocess`` (or by this module) is accepted directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from .config import TrainingConfig

DATASET_INFO = 'dataset_info.json'
DEFAULT_BATCH_SIZE = 16  # torch mace_run_train default (in structures)


@dataclass(frozen=True)
class PreparedData:
    """Locations the trainer reads plus the numbers derived from the training set."""

    train_dir: Path
    valid_dir: Path | None
    test_dir: Path | None
    statistics_file: Path
    batch_max_edges: int
    info: dict

    @property
    def hdf5_paths(self) -> dict[str, list[Path]]:
        out = {}
        for split, folder in (
            ('train', self.train_dir),
            ('valid', self.valid_dir),
            ('test', self.test_dir),
        ):
            if folder is None:
                continue
            out[split] = sorted(p for p in Path(folder).iterdir() if p.suffix in ('.h5', '.hdf5'))
        return out


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _is_hdf5_dir(paths: list[str]) -> bool:
    if len(paths) != 1:
        return False
    p = Path(paths[0])
    return p.is_dir() and any(c.suffix in ('.h5', '.hdf5') for c in p.iterdir())


def _fingerprint(config: TrainingConfig, files: dict[str, list[str]]) -> str:
    """Hash of the inputs and every option that changes the preprocessing output."""
    payload = {
        'files': {
            split: [
                (str(Path(p).resolve()), Path(p).stat().st_size, int(Path(p).stat().st_mtime))
                for p in paths
            ]
            for split, paths in files.items()
        },
        'r_max': config.r_max,
        'keys': [config.energy_key, config.forces_key, config.stress_key, config.virials_key, config.dipole_key],
        'E0s': config.E0s if not isinstance(config.E0s, dict) else sorted(config.E0s.items()),
        'atomic_numbers': config.atomic_numbers,
        'valid_fraction': config.valid_fraction if not files.get('valid') else None,
        'seed': config.seed,
        'scaling': config.scaling,
        'config_type_weights': sorted(config.config_type_weights.items()),
        'format': 2,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _preprocess_xyz(config: TrainingConfig, files: dict[str, list[str]], out_dir: Path) -> dict:
    """Run ``mace-jax-preprocess`` in-process and return its summary."""
    from mace_jax import tools  # noqa: PLC0415
    from mace_jax.cli import mace_jax_preprocess  # noqa: PLC0415

    argv = [
        '--train_file', json.dumps(files['train']),
        '--h5_prefix', str(out_dir),
        '--work_dir', str(out_dir),
        '--r_max', str(config.r_max),
        '--energy_key', config.energy_key,
        '--forces_key', config.forces_key,
        '--stress_key', config.stress_key,
        '--virials_key', config.virials_key,
        '--dipole_key', config.dipole_key,
        '--config_type_weights', json.dumps(config.config_type_weights),
        '--compute_statistics',
        '--scaling', config.scaling,
        '--seed', str(config.seed),
        '--valid_fraction', str(config.valid_fraction),
        '--num_process', str(max(1, min(8, config.num_workers or 1))),
        '--shuffle', str(bool(config.shuffle)),
    ]
    if files.get('valid'):
        argv += ['--valid_file', json.dumps(files['valid'])]
    if files.get('test'):
        argv += ['--test_file', json.dumps(files['test'])]
    if config.E0s is not None:
        argv += ['--E0s', config.E0s if isinstance(config.E0s, str) else str(config.E0s)]
    if config.atomic_numbers is not None:
        argv += ['--atomic_numbers', str([int(z) for z in config.atomic_numbers])]
    args = tools.build_preprocess_arg_parser().parse_args(argv)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = mace_jax_preprocess.run(args)
    return summary or {}


def _load_statistics(path: Path) -> dict:
    stats = json.loads(Path(path).read_text())
    for key in ('atomic_numbers', 'atomic_energies'):
        if isinstance(stats.get(key), str):
            import ast  # noqa: PLC0415

            stats[key] = ast.literal_eval(stats[key])
    return stats


def _batch_max_edges(config: TrainingConfig, stats: dict, info: dict) -> int:
    if config.batch_max_edges is not None:
        return int(config.batch_max_edges)
    batch_size = config.batch_size or DEFAULT_BATCH_SIZE
    mean_atoms = info.get('train', {}).get('mean_atoms')
    avg_nn = stats.get('avg_num_neighbors')
    if not mean_atoms or not avg_nn:
        edges = 8192
        logging.warning(
            'Dataset statistics do not give atoms per structure; using batch_max_edges=%d.', edges
        )
        return edges
    # directed edges per structure = atoms x neighbors per atom; 10 %% headroom so a batch of
    # `batch_size` typical structures fits the budget.
    edges = int(math.ceil(1.1 * batch_size * float(mean_atoms) * float(avg_nn)))
    return max(edges, 256)


def prepare_data(config: TrainingConfig, *, run_dir: Path | None = None) -> PreparedData:
    """XYZ (or HDF5) input -> HDF5 shards + statistics + the edge budget for batching."""
    run_dir = Path(run_dir) if run_dir is not None else config.run_dir
    files = {
        'train': _as_list(config.train_file),
        'valid': _as_list(config.valid_file),
        'test': _as_list(config.test_file),
    }
    for split, paths in files.items():
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f'{split}_file {p!r} does not exist')

    if _is_hdf5_dir(files['train']):
        train_dir = Path(files['train'][0])
        valid_dir = Path(files['valid'][0]) if files['valid'] else None
        test_dir = Path(files['test'][0]) if files['test'] else None
        stats_file = (
            Path(config.statistics_file)
            if config.statistics_file
            else train_dir.parent / 'statistics.json'
        )
        if not stats_file.exists():
            raise FileNotFoundError(
                f'HDF5 input needs a statistics file; none at {stats_file}. Run mace-jax-preprocess '
                'with --compute_statistics or pass statistics_file.'
            )
        info_path = train_dir.parent / DATASET_INFO
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
    else:
        for split, paths in files.items():
            for p in paths:
                if Path(p).suffix.lower() not in ('.xyz', '.extxyz'):
                    raise ValueError(
                        f'{split}_file {p!r}: give .xyz/.extxyz files or one directory of HDF5 shards'
                    )
        data_root = Path(config.data_dir) if config.data_dir else run_dir / 'data'
        key = _fingerprint(config, files)
        out_dir = data_root / key
        stats_file = Path(config.statistics_file) if config.statistics_file else out_dir / 'statistics.json'
        info_path = out_dir / DATASET_INFO
        if (out_dir / 'train').exists() and info_path.exists() and stats_file.exists():
            logging.info('Reusing preprocessed data in %s', out_dir)
            info = json.loads(info_path.read_text())
        else:
            logging.info('Preprocessing XYZ input into %s (r_max=%.3f)', out_dir, config.r_max)
            info = _preprocess_xyz(config, files, out_dir)
            info['source_files'] = files
            info_path.write_text(json.dumps(info, indent=1))
        train_dir = out_dir / 'train'
        valid_dir = out_dir / 'val' if (out_dir / 'val').exists() else None
        test_dir = out_dir / 'test' if (out_dir / 'test').exists() else None

    stats = _load_statistics(stats_file)
    edges = _batch_max_edges(config, stats, info)
    logging.info(
        'Data: %s | E0s for %s | avg_num_neighbors=%.3f | mean=%.6g std=%.6g | batch_max_edges=%d',
        train_dir,
        stats.get('atomic_numbers'),
        float(stats.get('avg_num_neighbors', float('nan'))),
        float(stats.get('mean', 0.0)),
        float(stats.get('std', 1.0)),
        edges,
    )
    return PreparedData(
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        statistics_file=Path(stats_file),
        batch_max_edges=edges,
        info=dict(info, statistics=stats),
    )


__all__ = ['PreparedData', 'prepare_data']
