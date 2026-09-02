"""`train(config)`: the Python entry point; the command line is a thin layer over it."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import TrainingConfig, describe
from .data import PreparedData, prepare_data
from .export import export_bundle


@dataclass(frozen=True)
class TrainingResult:
    """Where a finished run left its outputs."""

    run_dir: Path
    tag: str  # `<YYYYmmdd_HHMM>_<name>`, the prefix of every file the run wrote
    config_file: Path
    log_file: Path
    metrics_file: Path
    checkpoint_dir: Path
    best_checkpoint: Path | None
    bundle: Path | None  # the exported model (None if export_model=False)
    epochs: int
    final_metrics: dict  # last logged 'valid' (and 'eval_test') metrics, physical units
    data: PreparedData

    def summary(self) -> str:
        lines = [f'run {self.tag}: {self.epochs} epochs']
        for split in ('valid', 'eval_test'):
            m = self.final_metrics.get(split)
            if m:
                parts = []
                for key in ('rmse_e_per_atom', 'mae_e_per_atom', 'rmse_f', 'mae_f', 'loss'):
                    if key in m and m[key] is not None:
                        unit = ' meV/atom' if key.endswith('per_atom') else (' meV/A' if key.endswith('_f') else '')
                        value = 1000 * m[key] if unit else m[key]
                        parts.append(f'{key}={value:.4g}{unit}')
                lines.append(f'  {split}: ' + ', '.join(parts))
        if self.bundle:
            lines.append(f'  model: {self.bundle}')
        return '\n'.join(lines)


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = [p for p in checkpoint_dir.glob('*.ckpt') if not p.stem.endswith('_best')]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_metrics(path: Path) -> tuple[dict, int]:
    """Last metrics record per split from the run's JSON-lines .metrics file."""
    last: dict[str, dict] = {}
    epochs = 0
    if not path.exists():
        return last, epochs
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mode = record.get('mode')
            if mode:
                last[mode] = record
            if record.get('epoch') is not None:
                epochs = max(epochs, int(record['epoch']))
    return last, epochs


def train(config: TrainingConfig, *, print_config: bool = False) -> TrainingResult:
    """Train a model from ``config`` and return where everything landed.

    Steps: write the resolved config to the run directory; convert XYZ input to HDF5 shards +
    statistics (cached); translate the config into the gin-driven trainer's settings; run; export
    the final model as a bundle.
    """
    import gin  # noqa: PLC0415

    from mace_jax.cli import mace_jax_train as trainer_cli  # noqa: PLC0415

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = config.to_yaml(run_dir / f'{config.name}_config.yaml')

    if config.restart_latest and config.resume_from is None:
        latest = _latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            logging.info('restart_latest: resuming from %s', latest)
            config.resume_from = str(latest)
        else:
            logging.info('restart_latest: no checkpoint in %s, starting fresh', config.checkpoint_dir)

    data = prepare_data(config, run_dir=run_dir)
    argv = config.to_cli_args(data)
    if print_config:
        print(describe(config))
        print('# mace-jax-train arguments\n' + ' '.join(argv))

    # A fresh gin state per run so several train() calls in one process do not bleed into each other.
    gin.clear_config()
    args, remaining = trainer_cli.parse_args(argv)
    exit_code = trainer_cli._launch_local_processes(args)
    if exit_code is not None:
        raise SystemExit(exit_code)
    trainer_cli.configure_gin(args, remaining)
    trainer_cli.apply_cli_overrides(args)
    if print_config:
        print('# operative gin config\n' + gin.config_str())

    started = time.perf_counter()
    outcome = trainer_cli.run_training()
    elapsed = time.perf_counter() - started
    tag = outcome['tag']
    directory = Path(outcome['directory'])
    checkpoint_dir = Path(outcome.get('checkpoint_dir') or config.checkpoint_dir)
    best = checkpoint_dir / f'{tag}_best.ckpt'
    best_checkpoint = best if best.exists() else None

    bundle = None
    if config.export_model and outcome.get('eval_params') is not None:
        target = checkpoint_dir / (f'{tag}_les' if config.model == 'MACELES' else tag)
        bundle = export_bundle(
            (outcome['model_config'], outcome['eval_params']),
            target,
            extra_config={
                'source': {
                    'training_config': str(config_file.resolve()),
                    'best_checkpoint': str(best_checkpoint.resolve()) if best_checkpoint else None,
                    'statistics': str(data.statistics_file.resolve()),
                },
                'dtype': config.default_dtype,
            },
        )
        logging.info('Exported model bundle to %s', bundle)

    metrics_file = directory / f'{tag}.metrics'
    final_metrics, epochs = _read_metrics(metrics_file)
    result = TrainingResult(
        run_dir=directory,
        tag=tag,
        config_file=config_file,
        log_file=directory / f'{tag}.log',
        metrics_file=metrics_file,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=best_checkpoint,
        bundle=bundle,
        epochs=epochs,
        final_metrics=final_metrics,
        data=data,
    )
    logging.info('Training finished in %.1f s\n%s', elapsed, result.summary())
    return result


__all__ = ['TrainingResult', 'train']
