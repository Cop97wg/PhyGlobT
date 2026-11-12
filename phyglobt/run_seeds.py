"""Multi-seed training + evaluation + aggregation for PhyGlobT.

Runs the training pipeline once per seed, evaluates each resulting checkpoint on
the same evaluation scenes, and reports mean +/- std together with the per-scene
metrics needed for paired tests (see ``phyglobt.evaluate compare``).

Example (cloud, 3 seeds, evaluating the training script's own validation split)::

    python -m phyglobt.run_seeds \
        --data_dir data/processed \
        --out_root runs/us \
        --seeds 42 43 44 -- \
        --epochs 200 --batch_size 16

Pass ``--eval_dir`` to score a separate held-out corpus instead.

Everything after ``--`` is forwarded verbatim to ``phyglobt.train``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from .evaluate import run_eval


def _std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def main():
    p = argparse.ArgumentParser(description='Multi-seed PhyGlobT training and evaluation')
    p.add_argument('--data_dir', required=True, help='training scene directory')
    p.add_argument('--eval_dir', default=None,
                   help='held-out evaluation scene directory (defaults to --data_dir)')
    p.add_argument('--out_root', required=True, help='root for per-seed checkpoints/results')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--device', default=None)
    args, extra = p.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]
    eval_dir = args.eval_dir or args.data_dir
    # a separate eval_dir is used as-is; otherwise score the training script's
    # own 20% validation split so no training scene leaks into the report
    eval_split = 'all' if args.eval_dir else 'val'

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.out_root, exist_ok=True)
    per_seed = []

    for seed in args.seeds:
        save_dir = os.path.join(args.out_root, f'seed_{seed}')
        cmd = [sys.executable, '-m', 'phyglobt.train',
               '--data_dir', args.data_dir, '--save_dir', save_dir, '--seed', str(seed)] + extra
        print('\n>>>', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        ckpt = os.path.join(save_dir, 'best_model.pt')
        res = run_eval(eval_dir, ckpt, device, attention=False,
                       split=eval_split, split_seed=seed)
        out = os.path.join(save_dir, 'eval.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        per_seed.append(dict(seed=seed, eval=res, path=ckpt))
        print(f'seed {seed}: overall F1 = {res["overall"]["f1"]:.4f}')

    # ── aggregation ──
    summary = {'n_seeds': len(per_seed), 'seeds': args.seeds,
               'eval_dir': os.path.abspath(eval_dir), 'eval_split': eval_split}
    keys = ('precision', 'recall', 'f1')
    for key in keys:
        vals = [r['eval']['overall'][key] for r in per_seed]
        summary[key] = {'mean': float(np.mean(vals)), 'std': _std(vals), 'values': vals}
    bins = sorted({b for r in per_seed for b in r['eval']['by_density']})
    summary['by_density'] = {}
    for b in bins:
        vals = [r['eval']['by_density'].get(b, {}).get('f1') for r in per_seed]
        vals = [v for v in vals if v is not None]
        if vals:
            summary['by_density'][b] = {'mean': float(np.mean(vals)), 'std': _std(vals),
                                        'values': vals}
    summary['runs'] = [{'seed': r['seed'], 'checkpoint': os.path.abspath(r['path'])}
                       for r in per_seed]

    out = os.path.join(args.out_root, 'summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nOverall F1: {summary["f1"]["mean"]:.4f} +/- {summary["f1"]["std"]:.4f}')
    print(f'summary written: {out}')


if __name__ == '__main__':
    main()
