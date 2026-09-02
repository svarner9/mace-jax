"""``mace_jax_export``: write a mace-jax bundle from a training checkpoint, or describe a bundle.

    mace_jax_export results/checkpoints/20260902_1200_water_best.ckpt models/water
    mace_jax_export --describe models/water.npz

The bundle is ``<out>.npz`` + ``<out>.json`` for MACE / ScaleShiftMACE and the directory ``<out>/``
(``config.json`` + ``params.msgpack``) for MACE-LES: the formats `mace_jax.training.load_bundle`,
``mace-jax-predict`` and mlmd's ``MACE`` / ``MACELES`` calculators read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mace_jax.training.export import describe_bundle, export_bundle


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='mace_jax_export', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('source', help='training checkpoint (.ckpt), or a bundle with --describe')
    parser.add_argument('out', nargs='?', help='bundle path without extension (or directory for MACE-LES)')
    parser.add_argument(
        '--dtype', choices=['float32', 'float64'], default=None, help='store parameters in this precision'
    )
    parser.add_argument('--describe', action='store_true', help='print the architecture of SOURCE and exit')
    args = parser.parse_args(argv)
    if args.describe:
        print(describe_bundle(args.source))
        return
    if not args.out:
        parser.error('OUT is required unless --describe is given')
    path = export_bundle(args.source, args.out, dtype=args.dtype)
    print(f'wrote {path}')
    print(describe_bundle(path))


if __name__ == '__main__':
    main(sys.argv[1:])
