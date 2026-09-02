"""The training configuration: one dataclass, the torch-MACE argument names, YAML in and out.

`TrainingConfig` is the single source of truth for a run. It is built from Python keyword
arguments, from a YAML file (`TrainingConfig.from_yaml`), or from the command line
(`mace_jax_run_train`, whose parser is generated from the dataclass fields so the two never
drift). The trainer then translates it into what the underlying stack understands:
`to_cli_args()` produces the flags of the existing ``mace-jax-train`` entry point and
`to_gin_bindings()` the gin bindings for everything that entry point has no flag for (the model
architecture). Users never have to see either.

Field names follow torch ``mace_run_train`` where the concept exists in both codes, so a torch
command line can be reused with few edits. Differences are documented on the field.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

MODEL_CHOICES = ('MACE', 'ScaleShiftMACE', 'MACELES')
SCALING_CHOICES = ('rms_forces_scaling', 'std_scaling', 'no_scaling')
LOSS_CHOICES = (
    'weighted',
    'ef',
    'forces_only',
    'stress',
    'huber',
    'virials',
    'dipole',
    'energy_forces_dipole',
    'l1l2',
    'l1l2energyforces',
    'universal',
)
# Constructor arguments of the loss class behind each `loss` choice (mace_jax.modules.loss).
_EFS = ('energy_weight', 'forces_weight', 'stress_weight')
LOSS_PARAMETERS = {
    'weighted': _EFS,
    'stress': _EFS,
    'ef': ('energy_weight', 'forces_weight'),
    'forces_only': ('forces_weight',),
    'huber': _EFS + ('huber_delta',),
    'universal': _EFS + ('huber_delta',),
    'virials': ('energy_weight', 'forces_weight', 'virials_weight'),
    'dipole': ('energy_weight', 'forces_weight', 'dipole_weight'),
    'energy_forces_dipole': ('energy_weight', 'forces_weight', 'dipole_weight'),
    'l1l2': ('energy_weight', 'forces_weight'),
    'l1l2energyforces': ('energy_weight', 'forces_weight'),
}
OPTIMIZER_CHOICES = ('adam', 'adamw', 'amsgrad', 'sgd', 'schedulefree')
SCHEDULER_CHOICES = ('ReduceLROnPlateau', 'ExponentialLR', 'constant', 'plateau', 'exponential')
ERROR_TABLE_CHOICES = (
    'PerAtomRMSE',
    'rel_PerAtomRMSE',
    'TotalRMSE',
    'PerAtomMAE',
    'rel_PerAtomMAE',
    'TotalMAE',
    'PerAtomRMSEstressvirials',
    'PerAtomMAEstressvirials',
    'DipoleRMSE',
    'DipolePolarRMSE',
    'EnergyDipoleRMSE',
)
INTERACTION_CHOICES = (
    'RealAgnosticResidualInteractionBlock',
    'RealAgnosticAttResidualInteractionBlock',
    'RealAgnosticInteractionBlock',
    'RealAgnosticDensityInteractionBlock',
    'RealAgnosticDensityResidualInteractionBlock',
    'RealAgnosticResidualNonLinearInteractionBlock',
)
GATE_CHOICES = ('silu', 'tanh', 'abs', 'None')
DISTANCE_TRANSFORM_CHOICES = ('None', 'Agnesi', 'Soft')
RADIAL_TYPE_CHOICES = ('bessel', 'gaussian', 'chebyshev')


def _f(default, help: str, *, choices=None, cli: bool = True, group: str = 'other', **kw):
    """A dataclass field with the metadata the CLI generator and the docs read."""
    metadata = {'help': help, 'choices': choices, 'cli': cli, 'group': group}
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: default.copy(), metadata=metadata, **kw)
    return field(default=default, metadata=metadata, **kw)


@dataclass
class TrainingConfig:
    """Everything that defines a training run. Units: eV, eV/Å, Å.

    Torch users: the field names are the ``mace_run_train`` flags. The differences that matter:

    * ``batch_size`` here is a *target*: mace-jax packs structures into fixed-shape batches by an
      edge budget (``batch_max_edges``). If you give ``batch_size``, the budget is derived from the
      dataset statistics (mean atoms per structure × mean neighbors); setting ``batch_max_edges``
      directly overrides it.
    * ``scaling`` is applied inside the model (``ScaleShiftMACE``/``MACELES``, the default), so a
      saved model returns eV directly. ``model='MACE'`` keeps the older mace-jax behaviour of an
      external affine transform recorded as ``output_mean``/``output_std``.
    * ``default_dtype`` float32 runs with TF32 disabled (``matmul_precision='highest'``).
    * Checkpoints are pickled ``.ckpt`` files; the exported model is a mace-jax bundle
      (``<name>.npz`` + ``<name>.json``, or a directory for MACE-LES) that mace-jax and mlmd load.
    """

    # ---- run -----------------------------------------------------------------------------
    name: str = _f('mace', 'Run name; prefixes every output file.', group='run')
    results_dir: str = _f('results', 'Directory for logs, metrics, checkpoints and the exported model.', group='run')
    seed: int = _f(123, 'Random seed (parameters and data shuffling).', group='run')
    default_dtype: str = _f('float32', 'Numeric precision of the model and training.', choices=('float32', 'float64'), group='run')
    device: str | None = _f(None, "JAX platform: 'cpu', 'cuda', 'tpu' or None for JAX's default.", group='run')
    matmul_precision: str | None = _f('highest', "jax_default_matmul_precision. 'highest' turns TF32 off on NVIDIA GPUs (with TF32 on, MACE float32 forces were off by 1.6e-2 eV/A RMS). None keeps JAX's default.", group='run')
    log_level: str = _f('INFO', 'Python logging level.', choices=('DEBUG', 'INFO', 'WARNING'), group='run')
    debug: bool = _f(False, 'Enable JAX NaN/Inf checks (jax_debug_nans/infs); slow, for diagnosing a diverging run.', group='run')

    # ---- data ------------------------------------------------------------------------------
    train_file: str | list[str] | None = _f(None, 'Training set: .xyz/.extxyz file(s), or a directory of mace-jax HDF5 shards.', group='data')
    valid_file: str | list[str] | None = _f(None, 'Validation set (same formats). Omit to split valid_fraction off the training set.', group='data')
    test_file: str | list[str] | None = _f(None, 'Test set (same formats); evaluated at the end and, with eval_test, every eval_interval.', group='data')
    valid_fraction: float = _f(0.1, 'Fraction of the training set held out for validation when valid_file is not given.', group='data')
    energy_key: str = _f('energy', 'Name of the energy in the XYZ info fields.', group='data')
    forces_key: str = _f('forces', 'Name of the forces in the XYZ arrays.', group='data')
    stress_key: str = _f('stress', 'Name of the stress in the XYZ info fields (eV/A^3).', group='data')
    virials_key: str = _f('virials', 'Name of the virials in the XYZ info fields.', group='data')
    dipole_key: str = _f('dipole', 'Name of the dipole in the XYZ info fields.', group='data')
    config_type_weights: dict = _f({'Default': 1.0}, 'Per config_type loss weights.', group='data')
    E0s: str | dict | None = _f('average', "Isolated-atom reference energies: 'average' (least-squares fit over the training compositions), a {Z: E} dict, or a JSON file path. Isolated-atom frames in the training set (config_type=IsolatedAtom) take precedence.", group='data')
    atomic_numbers: list[int] | None = _f(None, 'Elements the model knows; default: the elements present in train + valid.', group='data')
    statistics_file: str | None = _f(None, 'Reuse an existing statistics.json (E0s, avg_num_neighbors, mean, std) instead of computing one.', group='data')
    data_dir: str | None = _f(None, 'Where the HDF5 conversion of the XYZ input is cached; default <results_dir>/data.', group='data')
    batch_size: int | None = _f(None, 'Target structures per batch; converted to an edge budget from the dataset statistics (see batch_max_edges).', group='data')
    valid_batch_size: int | None = _f(None, 'Unused (evaluation uses the training edge budget); accepted for torch compatibility.', group='data', cli=False)
    batch_max_edges: int | None = _f(None, 'Edge budget per padded batch (mace-jax batching unit). Overrides batch_size. Default: batch_size 16 worth of edges, or 8192.', group='data')
    num_workers: int = _f(0, 'Data-loading worker processes per training process.', group='data')
    shuffle: bool = _f(True, 'Shuffle the training set every epoch.', group='data')

    # ---- model -----------------------------------------------------------------------------
    model: str = _f('ScaleShiftMACE', "Model class. 'ScaleShiftMACE' (torch default) and 'MACELES' put the energy scale/shift inside the model; 'MACE' applies it outside (recorded as output_mean/output_std).", choices=MODEL_CHOICES, group='model')
    r_max: float = _f(5.0, 'Cutoff radius (A).', group='model')
    num_interactions: int = _f(2, 'Number of message-passing layers.', group='model')
    max_ell: int = _f(3, 'Highest spherical-harmonic degree of the edge features.', group='model')
    correlation: int = _f(3, 'Body order of the equivariant product basis minus one (3 = 4-body).', group='model')
    hidden_irreps: str | None = _f('128x0e + 128x1o', "Hidden features, e.g. '128x0e + 128x1o + 128x2e'. Overrides num_channels/max_L.", group='model')
    num_channels: int | None = _f(None, 'Torch shorthand: channels per irrep; with max_L builds hidden_irreps.', group='model')
    max_L: int | None = _f(None, 'Torch shorthand: highest l of the hidden features (with num_channels).', group='model')
    MLP_irreps: str = _f('16x0e', 'Hidden width of the nonlinear readout.', group='model')
    num_radial_basis: int = _f(8, 'Number of Bessel radial basis functions.', group='model')
    num_cutoff_basis: int = _f(5, 'Polynomial order of the radial cutoff envelope.', group='model')
    radial_type: str = _f('bessel', 'Radial basis type.', choices=RADIAL_TYPE_CHOICES, group='model')
    radial_MLP: list[int] = _f([64, 64, 64], 'Hidden layers of the radial MLP.', group='model')
    interaction: str = _f('RealAgnosticResidualInteractionBlock', 'Interaction block for layers 2..N.', choices=INTERACTION_CHOICES, group='model')
    interaction_first: str = _f('RealAgnosticResidualInteractionBlock', 'Interaction block for the first layer.', choices=INTERACTION_CHOICES, group='model')
    gate: str = _f('silu', 'Nonlinearity of the readout MLP.', choices=GATE_CHOICES, group='model')
    pair_repulsion: bool = _f(False, 'Add the ZBL pair-repulsion term.', group='model')
    distance_transform: str = _f('None', 'Radial distance transform.', choices=DISTANCE_TRANSFORM_CHOICES, group='model')
    use_reduced_cg: bool = _f(True, 'Reduced Clebsch-Gordan basis in the symmetric contraction (torch default True). Part of the model definition: never change it for a saved model.', group='model')
    scaling: str = _f('rms_forces_scaling', "Energy scale/shift: 'rms_forces_scaling' (scale = RMS force, shift = mean interaction energy per atom), 'std_scaling', 'no_scaling'.", choices=SCALING_CHOICES, group='model')
    avg_num_neighbors: float | None = _f(None, 'Override the average neighbor count used to normalize messages (default: from the data).', group='model')
    les_sigma: float | None = _f(None, 'MACELES: Gaussian width of the latent charges (A); default 1.0.', group='model')
    les_dl: float | None = _f(None, 'MACELES: reciprocal-space resolution (A); default 2.0.', group='model')
    les_norm_factor: float | None = _f(None, 'MACELES: LES energy prefactor; default 90.4756 (Coulomb constant in eV*A/e^2).', group='model')

    # ---- loss ------------------------------------------------------------------------------
    loss: str = _f('weighted', 'Loss function.', choices=LOSS_CHOICES, group='loss')
    energy_weight: float = _f(1.0, 'Weight of the energy term.', group='loss')
    forces_weight: float = _f(100.0, 'Weight of the forces term.', group='loss')
    stress_weight: float = _f(0.0, 'Weight of the stress term (needs stress labels).', group='loss')
    virials_weight: float = _f(0.0, 'Weight of the virials term.', group='loss')
    dipole_weight: float = _f(0.0, 'Weight of the dipole term.', group='loss')
    huber_delta: float = _f(0.01, 'Delta of the Huber losses.', group='loss')

    # ---- optimizer -------------------------------------------------------------------------
    optimizer: str = _f('adam', 'Optimizer.', choices=OPTIMIZER_CHOICES, group='optimizer')
    amsgrad: bool = _f(True, 'Use AMSGrad (torch default). Equivalent to optimizer=amsgrad.', group='optimizer')
    lr: float = _f(0.01, 'Learning rate.', group='optimizer')
    weight_decay: float = _f(5e-7, 'Weight decay on the interaction weights.', group='optimizer')
    beta: float = _f(0.9, 'Adam beta1.', group='optimizer')
    clip_grad: float | None = _f(10.0, 'Global gradient-norm clip; None disables.', group='optimizer')
    scheduler: str = _f('ReduceLROnPlateau', 'Learning-rate schedule.', choices=SCHEDULER_CHOICES, group='optimizer')
    lr_factor: float = _f(0.8, 'ReduceLROnPlateau: multiply the learning rate by this on a plateau.', group='optimizer')
    scheduler_patience: int = _f(50, 'ReduceLROnPlateau: epochs without validation improvement before reducing.', group='optimizer')
    lr_scheduler_gamma: float = _f(0.9993, 'ExponentialLR: per-epoch decay.', group='optimizer')
    ema: bool = _f(False, 'Evaluate and save an exponential moving average of the parameters.', group='optimizer')
    ema_decay: float = _f(0.99, 'EMA decay.', group='optimizer')
    max_num_epochs: int = _f(2048, 'Maximum number of epochs.', group='optimizer')
    patience: int = _f(2048, 'Early stopping: epochs without validation improvement.', group='optimizer')
    eval_interval: int = _f(1, 'Validate every N epochs.', group='optimizer')
    eval_train: bool = _f(True, 'Also report training-set metrics every eval_interval.', group='optimizer')
    eval_test: bool = _f(False, 'Also report test-set metrics every eval_interval (always reported at the end).', group='optimizer')

    # ---- stage two / SWA --------------------------------------------------------------------
    swa: bool = _f(False, 'Stage two: at start_swa switch to the swa_* loss weights and swa_lr, and average parameters.', group='swa')
    start_swa: int | None = _f(None, 'Epoch at which stage two starts; default 75%% of max_num_epochs.', group='swa')
    swa_lr: float = _f(1e-3, 'Learning rate in stage two.', group='swa')
    swa_energy_weight: float = _f(1000.0, 'Energy weight in stage two.', group='swa')
    swa_forces_weight: float = _f(100.0, 'Forces weight in stage two.', group='swa')
    swa_stress_weight: float = _f(10.0, 'Stress weight in stage two.', group='swa')
    swa_virials_weight: float = _f(10.0, 'Virials weight in stage two.', group='swa')
    swa_dipole_weight: float = _f(1.0, 'Dipole weight in stage two.', group='swa')

    # ---- checkpoints -----------------------------------------------------------------------
    checkpoint_every: int = _f(10, 'Write a resumable checkpoint every N epochs (0 disables; the best model is always saved).', group='checkpoints')
    keep_checkpoints: int = _f(2, 'How many periodic checkpoints to keep (0 = all).', group='checkpoints')
    save_all_checkpoints: bool = _f(False, 'Keep every periodic checkpoint (same as keep_checkpoints=0).', group='checkpoints')
    restart_latest: bool = _f(False, 'Resume from the newest checkpoint in <results_dir>/checkpoints if there is one.', group='checkpoints')
    resume_from: str | None = _f(None, 'Resume from this checkpoint file.', group='checkpoints')
    export_model: bool = _f(True, 'Export the final model as a mace-jax bundle next to the checkpoints.', group='checkpoints')

    # ---- logging ---------------------------------------------------------------------------
    error_table: str = _f('PerAtomRMSE', 'Which error summary to print each evaluation.', choices=ERROR_TABLE_CHOICES, group='logging')
    wandb: bool = _f(False, 'Log to Weights & Biases.', group='logging')
    wandb_project: str | None = _f(None, 'W&B project.', group='logging')
    wandb_entity: str | None = _f(None, 'W&B entity.', group='logging')
    wandb_name: str | None = _f(None, 'W&B run name (default: the run tag).', group='logging')
    wandb_dir: str | None = _f(None, 'W&B local directory.', group='logging')

    # ---- acceleration / parallelism ------------------------------------------------------------
    enable_cueq: bool = _f(False, 'Use the cuEquivariance kernels (needs cuequivariance-ops-jax).', group='runtime')
    launcher: str = _f('none', "Multi-GPU launcher: 'none' (one device), 'local' (one process per visible GPU), 'auto'.", choices=('none', 'local', 'auto'), group='runtime')
    distributed: bool = _f(False, 'Initialize jax.distributed (multi-host); see docs/training.md.', group='runtime')

    # ---- escape hatches --------------------------------------------------------------------
    gin_configs: list[str] = _f([], 'gin config files applied before the fields above (advanced).', group='advanced')
    gin_bindings: list[str] = _f([], "gin bindings applied last, e.g. \"mace_jax.tools.gin_functions.train.log_errors='PerAtomMAE'\" (advanced).", group='advanced')

    # ------------------------------------------------------------------------------------------
    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.model not in MODEL_CHOICES:
            raise ValueError(f'model must be one of {MODEL_CHOICES}, got {self.model!r}')
        if self.train_file is None:
            raise ValueError('train_file is required')
        if self.hidden_irreps is None:
            if self.num_channels is None or self.max_L is None:
                raise ValueError('give hidden_irreps, or both num_channels and max_L')
            self.hidden_irreps = hidden_irreps_from_channels(self.num_channels, self.max_L)
        elif self.num_channels is not None or self.max_L is not None:
            raise ValueError('give either hidden_irreps or num_channels + max_L, not both')
        if not 0.0 < self.valid_fraction < 1.0:
            raise ValueError('valid_fraction must be in (0, 1)')
        if self.scheduler not in SCHEDULER_CHOICES:
            raise ValueError(f'scheduler must be one of {SCHEDULER_CHOICES}')
        if self.swa and self.start_swa is None:
            self.start_swa = int(0.75 * self.max_num_epochs)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if isinstance(self.E0s, str) and self.E0s.strip().startswith('{'):
            self.E0s = {int(k): float(v) for k, v in ast.literal_eval(self.E0s).items()}
        if isinstance(self.E0s, dict):
            self.E0s = {int(k): float(v) for k, v in self.E0s.items()}
        if self.save_all_checkpoints:
            self.keep_checkpoints = 0
        if self.model != 'MACELES' and any(
            v is not None for v in (self.les_sigma, self.les_dl, self.les_norm_factor)
        ):
            raise ValueError('les_* options only apply to model="MACELES"')

    # ---- serialization -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TrainingConfig':
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f'unknown TrainingConfig options: {unknown}')
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'TrainingConfig':
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f'{path} must hold a mapping of option: value')
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> Path:
        import yaml  # noqa: PLC0415

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    # ---- derived quantities ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir)

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / 'checkpoints'

    def effective_optimizer(self) -> str:
        if self.optimizer == 'adam' and self.amsgrad:
            return 'amsgrad'
        return self.optimizer

    def effective_scheduler(self) -> str:
        return {
            'ReduceLROnPlateau': 'plateau',
            'ExponentialLR': 'exponential',
        }.get(self.scheduler, self.scheduler)

    # ---- translation to the underlying stack ---------------------------------------------------
    def to_cli_args(self, data) -> list[str]:
        """Flags for ``mace_jax.cli.mace_jax_train`` (everything that entry point supports).

        ``data`` is the `mace_jax.training.data.PreparedData` describing the HDF5 shards and the
        statistics file the trainer reads.
        """
        args: list[str] = []

        def add(flag, value):
            if value is None:
                return
            args.extend([flag, str(value)])

        add('--name', self.name)
        add('--log-dir', str(self.run_dir))
        add('--seed', self.seed)
        add('--dtype', self.default_dtype)
        add('--device', self.device)
        add('--train-file', str(data.train_dir))
        add('--valid-file', str(data.valid_dir) if data.valid_dir else 'None')
        add('--test-file', str(data.test_dir) if data.test_dir else 'None')
        add('--statistics-file', str(data.statistics_file))
        add('--r-max', self.r_max)
        add('--energy-key', self.energy_key)
        add('--forces-key', self.forces_key)
        add('--batch-max-edges', data.batch_max_edges)
        add('--num-workers', self.num_workers)
        add('--suffle', self.shuffle)  # the underlying flag is spelled this way
        add('--loss', self.loss if self.loss != 'universal' else None)
        # Each loss class accepts only its own weights; passing another one is an error.
        accepted = LOSS_PARAMETERS[self.loss]
        if 'energy_weight' in accepted:
            add('--energy-weight', self.energy_weight)
        if 'forces_weight' in accepted:
            add('--forces-weight', self.forces_weight)
        if 'stress_weight' in accepted:
            add('--stress-weight', self.stress_weight)
        if 'virials_weight' in accepted:
            add('--virials-weight', self.virials_weight)
        if 'dipole_weight' in accepted:
            add('--dipole-weight', self.dipole_weight)
        if 'huber_delta' in accepted:
            add('--huber-delta', self.huber_delta)
        add('--optimizer', self.effective_optimizer())
        add('--lr', self.lr)
        add('--weight-decay', self.weight_decay)
        add('--beta', self.beta)
        add('--clip-grad', self.clip_grad)
        add('--scheduler', self.effective_scheduler())
        add('--lr-factor', self.lr_factor)
        add('--scheduler-patience', self.scheduler_patience)
        add('--lr-scheduler-gamma', self.lr_scheduler_gamma)
        if self.ema:
            args.append('--ema')
            add('--ema-decay', self.ema_decay)
        add('--max-epochs', self.max_num_epochs)
        add('--patience', self.patience)
        add('--eval-interval', self.eval_interval)
        args.append('--eval-train' if self.eval_train else '--no-eval-train')
        args.append('--eval-test' if self.eval_test else '--no-eval-test')
        add('--log-errors', self.error_table)
        if self.swa:
            args.append('--swa')
            add('--swa-start', self.start_swa)
            add('--swa-lr', self.swa_lr)
            if 'energy_weight' in accepted:
                add('--swa-energy-weight', self.swa_energy_weight)
            if 'forces_weight' in accepted:
                add('--swa-forces-weight', self.swa_forces_weight)
            if 'stress_weight' in accepted:
                add('--swa-stress-weight', self.swa_stress_weight)
            if 'virials_weight' in accepted:
                add('--swa-virials-weight', self.swa_virials_weight)
            if 'dipole_weight' in accepted:
                add('--swa-dipole-weight', self.swa_dipole_weight)
        add('--checkpoint-dir', str(self.checkpoint_dir))
        if self.checkpoint_every and self.checkpoint_every > 0:
            add('--checkpoint-every', self.checkpoint_every)
            add('--checkpoint-keep', self.keep_checkpoints)
        args.append('--checkpoint-best')
        add('--resume-from', self.resume_from)
        if self.wandb:
            args.append('--wandb')
            add('--wandb-project', self.wandb_project)
            add('--wandb-entity', self.wandb_entity)
            add('--wandb-run-name', self.wandb_name)
            add('--wandb-dir', self.wandb_dir)
        if self.enable_cueq:
            args.append('--enable-cueq')
        add('--launcher', self.launcher)
        if self.distributed:
            args.append('--distributed')
        for binding in self.to_gin_bindings():
            args.extend(['--binding', binding])
        args.extend(self.gin_configs)
        return args

    def to_gin_bindings(self) -> list[str]:
        """gin bindings for the model architecture (no ``mace-jax-train`` flag exists for these)."""
        m = 'mace_jax.tools.gin_model.model'
        b = [
            f"{m}.model_type = '{self.model}'",
            f'{m}.num_interactions = {int(self.num_interactions)}',
            f'{m}.max_ell = {int(self.max_ell)}',
            f'{m}.correlation = {int(self.correlation)}',
            f"{m}.hidden_irreps = '{self.hidden_irreps}'",
            f"{m}.MLP_irreps = '{self.MLP_irreps}'",
            f'{m}.num_bessel = {int(self.num_radial_basis)}',
            f'{m}.num_polynomial_cutoff = {int(self.num_cutoff_basis)}',
            f"{m}.radial_type = '{self.radial_type}'",
            f'{m}.radial_MLP = {list(int(x) for x in self.radial_MLP)}',
            f'{m}.interaction_cls = @mace_jax.modules.{self.interaction}',
            f'{m}.interaction_cls_first = @mace_jax.modules.{self.interaction_first}',
            f'{m}.gate = ' + ('None' if self.gate == 'None' else f'@{self.gate}'),
            f'{m}.pair_repulsion = {bool(self.pair_repulsion)}',
            f"{m}.distance_transform = '{self.distance_transform}'",
            f'{m}.use_reduced_cg = {bool(self.use_reduced_cg)}',
            f"mace_jax.tools.gin_functions.flags.matmul_precision = "
            + ('None' if self.matmul_precision is None else f"'{self.matmul_precision}'"),
            f'mace_jax.tools.gin_functions.flags.debug = {bool(self.debug)}',
        ]
        if self.avg_num_neighbors is not None:
            b.append(f'{m}.avg_num_neighbors = {float(self.avg_num_neighbors)}')
        if self.loss == 'universal':
            b.append('mace_jax.tools.gin_functions.loss.loss_cls = @mace_jax.modules.UniversalLoss')
        if self.model == 'MACELES':
            for key in ('les_sigma', 'les_dl', 'les_norm_factor'):
                value = getattr(self, key)
                if value is not None:
                    b.append(f'{m}.{key} = {float(value)}')
        b.extend(self.gin_bindings)
        return b


def hidden_irreps_from_channels(num_channels: int, max_L: int) -> str:
    """torch's ``--num_channels/--max_L`` shorthand: ``128x0e + 128x1o + 128x2e`` for (128, 2)."""
    parts = [f'{num_channels}x{l}{"e" if l % 2 == 0 else "o"}' for l in range(int(max_L) + 1)]
    return ' + '.join(parts)


# ---- command-line parser generated from the dataclass -------------------------------------------

_LIST_INT = re.compile(r'^\[?\s*(-?\d+\s*(,\s*-?\d+\s*)*)?\]?$')


def _parse_value(text: str, annotation: str):
    """Parse a CLI string into the field's type (annotations are strings under `from __future__`)."""
    if text is None:
        return None
    if text.strip().lower() in {'none', 'null'} and 'None' in annotation:
        return None
    if annotation.startswith('bool'):
        low = text.strip().lower()
        if low in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if low in {'0', 'false', 'no', 'n', 'off'}:
            return False
        raise argparse.ArgumentTypeError(f'expected a boolean, got {text!r}')
    if annotation.startswith('int'):
        return int(text)
    if annotation.startswith('float'):
        return float(text)
    if annotation.startswith('list[int]'):
        if not _LIST_INT.match(text):
            raise argparse.ArgumentTypeError(f'expected integers like "64,64,64", got {text!r}')
        return [int(x) for x in text.strip('[] ').split(',') if x.strip()]
    if annotation.startswith('list[str]'):
        return [x.strip() for x in text.split(',') if x.strip()]
    if annotation.startswith('dict'):
        return ast.literal_eval(text)
    if annotation.startswith('str | dict'):
        return text
    if annotation.startswith('str | list'):
        parts = [x.strip() for x in text.split(',') if x.strip()]
        return parts[0] if len(parts) == 1 else parts
    if annotation.startswith('list[int] | None'):
        return [int(x) for x in text.strip('[] ').split(',') if x.strip()]
    return text


def build_arg_parser(prog: str = 'mace_jax_run_train') -> argparse.ArgumentParser:
    """An argparse parser with one ``--option`` per `TrainingConfig` field (torch names)."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Train a MACE or MACE-LES model in JAX. Options mirror torch mace_run_train; '
        'any option may also come from a YAML file given with --config (command-line values win).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', help='YAML file with TrainingConfig options.', default=None)
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print the resolved configuration (and the gin bindings it becomes) and exit.',
    )
    groups: dict[str, argparse._ArgumentGroup] = {}
    for f in fields(TrainingConfig):
        meta = f.metadata
        if not meta.get('cli', True):
            continue
        group = groups.setdefault(meta['group'], parser.add_argument_group(meta['group']))
        annotation = str(f.type)
        kwargs: dict[str, Any] = {
            'dest': f.name,
            'default': argparse.SUPPRESS,
            'help': meta['help'],
            'type': lambda s, a=annotation: _parse_value(s, a),
        }
        if meta.get('choices'):
            kwargs['choices'] = list(meta['choices'])
            kwargs.pop('type')
        if annotation.startswith('bool'):
            # `--ema` and `--ema true/false` both work; `--no-ema` too.
            kwargs.update(nargs='?', const=True)
            group.add_argument(f'--{f.name}', **kwargs)
            group.add_argument(
                f'--no-{f.name}',
                dest=f.name,
                action='store_const',
                const=False,
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )
            continue
        if annotation.startswith('list[str]') or annotation.startswith('str | list'):
            kwargs.update(nargs='+')
            kwargs['type'] = str
        flag_names = [f'--{f.name}']
        if '_' in f.name:
            flag_names.append('--' + f.name.replace('_', '-'))
        group.add_argument(*flag_names, **kwargs)
    return parser


def config_from_args(argv: list[str] | None = None) -> tuple[TrainingConfig, argparse.Namespace]:
    """Parse the command line (plus an optional ``--config`` YAML) into a `TrainingConfig`."""
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    values: dict[str, Any] = {}
    if ns.config:
        import yaml  # noqa: PLC0415

        values.update(yaml.safe_load(Path(ns.config).read_text()) or {})
    for f in fields(TrainingConfig):
        if hasattr(ns, f.name):
            value = getattr(ns, f.name)
            if (
                f.name in ('train_file', 'valid_file', 'test_file')
                and isinstance(value, list)
                and len(value) == 1
            ):
                value = value[0]
            values[f.name] = value
    return TrainingConfig.from_dict(values), ns


def describe(config: TrainingConfig) -> str:
    """The resolved configuration as YAML plus the gin bindings it implies (for --print-config)."""
    import yaml  # noqa: PLC0415

    text = yaml.safe_dump(config.to_dict(), sort_keys=False)
    bindings = '\n'.join(config.to_gin_bindings())
    return f'# TrainingConfig\n{text}\n# gin bindings\n{bindings}\n'


__all__ = [
    'TrainingConfig',
    'build_arg_parser',
    'config_from_args',
    'describe',
    'hidden_irreps_from_channels',
]

# JSON is handy for logging the config inside the run directory.
def config_to_json(config: TrainingConfig) -> str:
    return json.dumps(config.to_dict(), indent=1, sort_keys=False)
