"""End-to-end tests of the training API and CLI on tiny synthetic datasets (CPU, a few epochs).

Reference labels come from ASE's EMT potential on small Cu/Al clusters and periodic Cu cells, so
the tests need no data files. Models are made deliberately small; what is checked is the
plumbing: XYZ -> HDF5 + statistics, config -> gin, training runs and the loss falls, checkpoints and
bundles are written in the documented formats, a bundle loads in isolation and gives finite energies
and forces consistent with finite differences, MACE-LES batches carry a k-grid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('ase')
pytest.importorskip('yaml')

import ase.io  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402

from mace_jax.training import TrainingConfig, export_bundle, load_bundle, train  # noqa: E402
from mace_jax.training.config import config_from_args, hidden_irreps_from_channels  # noqa: E402
from mace_jax.training.export import ase_calculator  # noqa: E402

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

TINY_MODEL = dict(
    r_max=4.0,
    num_interactions=1,
    max_ell=1,
    correlation=2,
    hidden_irreps='8x0e + 8x1o',
    MLP_irreps='8x0e',
    num_radial_basis=4,
    num_cutoff_basis=3,
)


def _label(atoms: Atoms) -> Atoms:
    atoms.calc = EMT()
    atoms.info['energy'] = float(atoms.get_potential_energy())
    atoms.arrays['forces'] = np.asarray(atoms.get_forces())
    atoms.calc = None
    return atoms


def _cluster_frames(n: int, seed: int) -> list[Atoms]:
    """Cu/Al clusters of 6 atoms around a distorted octahedron."""
    rng = np.random.default_rng(seed)
    base = 2.5 * np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
    )
    frames = []
    for _ in range(n):
        symbols = ['Cu'] * 6
        for k in rng.choice(6, size=2, replace=False):
            symbols[k] = 'Al'
        atoms = Atoms(symbols, positions=base + rng.normal(scale=0.15, size=base.shape))
        atoms.center(vacuum=5.0)
        atoms.pbc = False
        frames.append(_label(atoms))
    return frames


def _periodic_frames(n: int, seed: int) -> list[Atoms]:
    """8-atom fcc Cu cells with random displacements (periodic)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat((2, 1, 1))
        atoms.positions += rng.normal(scale=0.08, size=atoms.positions.shape)
        frames.append(_label(atoms))
    return frames


@pytest.fixture(scope='module')
def cluster_xyz(tmp_path_factory):
    root = tmp_path_factory.mktemp('clusters')
    ase.io.write(root / 'train.xyz', _cluster_frames(40, seed=0))
    ase.io.write(root / 'valid.xyz', _cluster_frames(8, seed=1))
    ase.io.write(root / 'test.xyz', _cluster_frames(6, seed=2))
    return root


@pytest.fixture(scope='module')
def periodic_xyz(tmp_path_factory):
    root = tmp_path_factory.mktemp('periodic')
    ase.io.write(root / 'train.xyz', _periodic_frames(16, seed=0))
    ase.io.write(root / 'valid.xyz', _periodic_frames(4, seed=1))
    return root


def _finite_difference_forces(atoms: Atoms, calc, h: float = 1e-3) -> np.ndarray:
    forces = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for k in range(3):
            plus = atoms.copy()
            plus.positions[i, k] += h
            plus.calc = calc
            minus = atoms.copy()
            minus.positions[i, k] -= h
            minus.calc = calc
            forces[i, k] = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * h)
    return forces


# ---- configuration -------------------------------------------------------------------------------


def test_config_defaults_follow_torch_and_round_trip(tmp_path):
    config = TrainingConfig(train_file='a.xyz', results_dir=str(tmp_path))
    assert config.model == 'ScaleShiftMACE'
    assert config.effective_optimizer() == 'amsgrad'  # adam + amsgrad flag, like torch
    assert config.effective_scheduler() == 'plateau'
    path = config.to_yaml(tmp_path / 'c.yaml')
    again = TrainingConfig.from_yaml(path)
    assert again.to_dict() == config.to_dict()
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({'train_file': 'a.xyz', 'not_an_option': 1})


def test_config_from_command_line_uses_torch_names():
    config, ns = config_from_args(
        [
            '--train_file', 'a.xyz',
            '--num_channels', '32', '--max_L', '2', '--hidden_irreps', 'None',
            '--E0s', '{29: -1.0, 13: -2.0}',
            '--swa', '--max_num_epochs', '100',
            '--radial_MLP', '16,16',
            '--no-eval_train',
            '--gin_bindings', "mace_jax.tools.gin_functions.train.log_errors='PerAtomMAE'",
        ]
    )
    assert config.hidden_irreps == hidden_irreps_from_channels(32, 2) == '32x0e + 32x1o + 32x2e'
    assert config.E0s == {29: -1.0, 13: -2.0}
    assert config.swa and config.start_swa == 75
    assert config.radial_MLP == [16, 16]
    assert config.eval_train is False
    bindings = config.to_gin_bindings()
    assert "mace_jax.tools.gin_model.model.hidden_irreps = '32x0e + 32x1o + 32x2e'" in bindings
    assert bindings[-1].endswith("log_errors='PerAtomMAE'")


def test_config_rejects_conflicting_architecture_shorthand():
    with pytest.raises(ValueError):
        TrainingConfig(train_file='a.xyz', hidden_irreps='16x0e', num_channels=16, max_L=0)


# ---- training ------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def trained_scaleshift(cluster_xyz, tmp_path_factory):
    results = tmp_path_factory.mktemp('run_scaleshift')
    config = TrainingConfig(
        name='tiny',
        results_dir=str(results),
        train_file=str(cluster_xyz / 'train.xyz'),
        valid_file=str(cluster_xyz / 'valid.xyz'),
        test_file=str(cluster_xyz / 'test.xyz'),
        model='ScaleShiftMACE',
        batch_size=8,
        max_num_epochs=4,
        lr=0.02,
        ema=False,
        eval_interval=1,
        checkpoint_every=2,
        device='cpu',
        **TINY_MODEL,
    )
    return config, train(config)


def test_scaleshift_run_writes_everything(trained_scaleshift):
    config, result = trained_scaleshift
    assert result.log_file.exists() and result.metrics_file.exists()
    assert result.config_file.exists()
    assert result.best_checkpoint is not None and result.best_checkpoint.exists()
    assert result.bundle is not None and result.bundle.suffix == '.npz'
    assert result.bundle.with_suffix('.json').exists()
    # the XYZ input was converted once, with statistics over the training set
    stats = json.loads(result.data.statistics_file.read_text())
    assert set(json.loads(stats['atomic_numbers'].replace("'", '"'))) == {13, 29}
    assert result.data.info['train']['n_structures'] == 40
    assert result.epochs >= 3
    valid = result.final_metrics['valid']
    assert np.isfinite(valid['loss'])


def test_loss_decreases_over_epochs(trained_scaleshift):
    _, result = trained_scaleshift
    losses = []
    with result.metrics_file.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('mode') == 'valid':
                losses.append(float(record['loss']))
    assert len(losses) >= 3
    assert losses[-1] < losses[0]


def test_bundle_config_records_the_model_definition(trained_scaleshift):
    _, result = trained_scaleshift
    config = json.loads(result.bundle.with_suffix('.json').read_text())
    assert config['model_class'] == 'ScaleShiftMACE'
    assert config['torch_model_class'] == 'ScaleShiftMACE'
    assert 'atomic_inter_scale' in config and 'atomic_inter_shift' in config
    assert config['use_reduced_cg'] is True
    # the activation-normalization constant the model was built with (see docs/bundles.md)
    assert abs(config['normalize2mom_consts']['silu'] - 1.6742560863) < 1e-6
    # scale/shift live inside the model, so the external transform is the identity
    assert float(config['output_mean']) == 0.0 and float(config['output_std']) == 1.0
    assert config['source']['statistics'].endswith('statistics.json')


def test_bundle_loads_standalone_and_forces_match_finite_differences(trained_scaleshift, cluster_xyz):
    _, result = trained_scaleshift
    calc = ase_calculator(result.bundle, dtype='float64')
    atoms = ase.io.read(cluster_xyz / 'test.xyz', index=0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy) and np.all(np.isfinite(forces))
    fd = _finite_difference_forces(atoms, calc)
    assert np.abs(forces - fd).max() < 2e-4, np.abs(forces - fd).max()


def _assert_same_predictions(path_a, path_b, atoms, tol=1e-8):
    """Two model files must give the same energy and forces on `atoms` (float64)."""
    atoms = atoms.copy()
    atoms.calc = ase_calculator(path_a, dtype='float64')
    e1, f1 = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = ase_calculator(path_b, dtype='float64')
    e2, f2 = atoms.get_potential_energy(), atoms.get_forces()
    assert np.isfinite(e1) and abs(e1 - e2) < tol, (e1, e2)
    assert np.abs(f1 - f2).max() < tol


def test_bundle_matches_checkpoint_predictions(trained_scaleshift, cluster_xyz, tmp_path):
    """A bundle exported from a checkpoint is the same model as the checkpoint (guards against
    writing initial parameters instead of the loaded ones)."""
    _, result = trained_scaleshift
    atoms = ase.io.read(cluster_xyz / 'test.xyz', index=1)
    exported = export_bundle(result.best_checkpoint, tmp_path / 'from_best')
    _assert_same_predictions(exported, result.best_checkpoint, atoms)
    # and the run's own export equals its final evaluation parameters (last epoch checkpoint)
    last = sorted(result.checkpoint_dir.glob('*_epoch*.ckpt'))[-1]
    _assert_same_predictions(result.bundle, last, atoms)


def test_export_cli_rewrites_a_checkpoint(trained_scaleshift, tmp_path):
    _, result = trained_scaleshift
    out = export_bundle(result.best_checkpoint, tmp_path / 'exported')
    assert out.exists() and out.with_suffix('.json').exists()
    bundle = load_bundle(out, 'float32')
    assert bundle.config['model_class'] == 'ScaleShiftMACE'


def test_rerun_reuses_preprocessed_data(trained_scaleshift):
    config, result = trained_scaleshift
    from mace_jax.training.data import prepare_data

    again = prepare_data(config)
    assert again.train_dir == result.data.train_dir
    assert again.batch_max_edges == result.data.batch_max_edges


def test_plain_mace_records_external_normalization(cluster_xyz, tmp_path):
    config = TrainingConfig(
        name='plain',
        results_dir=str(tmp_path),
        train_file=str(cluster_xyz / 'train.xyz'),
        valid_file=str(cluster_xyz / 'valid.xyz'),
        model='MACE',
        batch_size=8,
        max_num_epochs=1,
        device='cpu',
        **TINY_MODEL,
    )
    result = train(config)
    meta = json.loads(result.bundle.with_suffix('.json').read_text())
    stats = json.loads(result.data.statistics_file.read_text())
    assert meta['model_class'] == 'MACE'
    assert abs(float(meta['output_std']) - float(stats['std'])) < 1e-9
    assert abs(float(meta['output_mean']) - float(stats['mean'])) < 1e-9
    calc = ase_calculator(result.bundle, dtype='float64')
    atoms = ase.io.read(cluster_xyz / 'valid.xyz', index=0)
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())


def test_run_train_cli_print_config(cluster_xyz, capsys):
    from mace_jax.cli import mace_jax_run_train

    mace_jax_run_train.main(
        ['--train_file', str(cluster_xyz / 'train.xyz'), '--hidden_irreps', '8x0e', '--print-config']
    )
    out = capsys.readouterr().out
    assert 'model: ScaleShiftMACE' in out
    assert "mace_jax.tools.gin_model.model.hidden_irreps = '8x0e'" in out


# ---- MACE-LES ------------------------------------------------------------------------------------


def test_attach_kgrid_pads_with_inert_rows():
    import jraph

    from les_jax.ewald import build_kgrid
    from mace_jax.data.utils import config_from_atoms, graph_from_configuration
    from mace_jax.training.les import attach_kgrid

    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    graph = graph_from_configuration(config_from_atoms(atoms), cutoff=4.0)
    padded = jraph.pad_with_graphs(
        graph, n_node=int(graph.n_node.sum()) + 2, n_edge=int(graph.n_edge.sum()) + 4, n_graph=2
    )
    grid = build_kgrid(atoms.cell.array, dl=2.0)
    out = attach_kgrid(padded, num_k=grid.shape[0] + 5, dl=2.0)
    nvec = np.asarray(out.globals.les_nvec)
    assert nvec.shape == (2, grid.shape[0] + 5, 3)
    np.testing.assert_array_equal(nvec[0, : grid.shape[0]], grid)
    assert not nvec[0, grid.shape[0] :].any()
    assert not nvec[1].any()  # the padding graph has no reciprocal-space term
    with pytest.raises(ValueError):
        attach_kgrid(padded, num_k=grid.shape[0] - 1, dl=2.0)


def test_maceles_trains_on_periodic_data_and_exports_directory_bundle(periodic_xyz, tmp_path):
    config = TrainingConfig(
        name='les',
        results_dir=str(tmp_path),
        train_file=str(periodic_xyz / 'train.xyz'),
        valid_file=str(periodic_xyz / 'valid.xyz'),
        model='MACELES',
        batch_size=4,
        max_num_epochs=2,
        checkpoint_every=1,
        device='cpu',
        **TINY_MODEL,
    )
    result = train(config)
    assert result.bundle is not None and result.bundle.is_dir()
    assert (result.bundle / 'config.json').exists() and (result.bundle / 'params.msgpack').exists()
    meta = json.loads((result.bundle / 'config.json').read_text())
    assert meta['model_class'] == 'LESMACE'
    assert meta['les'] == {'sigma': 1.0, 'dl': 2.0, 'remove_self_interaction': True, 'norm_factor': 90.4756}
    assert 'atomic_inter_scale' in meta
    # loads through the mlmd MACELES path (config.json + Param-only msgpack) and evaluates
    calc = ase_calculator(result.bundle, dtype='float64')
    atoms = ase.io.read(periodic_xyz / 'valid.xyz', index=0)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy) and np.all(np.isfinite(forces))
    fd = _finite_difference_forces(atoms, calc)
    assert np.abs(forces - fd).max() < 5e-4, np.abs(forces - fd).max()
    # the reciprocal-space Ewald term is active: without the k-grid the model would silently fall
    # back to the real-space branch and give a different energy
    assert calc.graph_transform is not None
    calc.graph_transform = None
    calc.results = {}
    atoms.calc = calc
    energy_without_kgrid = atoms.get_potential_energy()
    assert abs(energy_without_kgrid - energy) > 1e-6
    # the directory bundle holds the trained parameters, not the initial ones
    last = sorted(result.checkpoint_dir.glob('*_epoch*.ckpt'))[-1]
    _assert_same_predictions(result.bundle, last, atoms)
    reexported = export_bundle(result.best_checkpoint, tmp_path / 'les_from_best')
    _assert_same_predictions(reexported, result.best_checkpoint, atoms)
