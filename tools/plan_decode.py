"""Rail the Phy-Sinkhorn plan back into the decision and compare it with the
per-pair threshold rule, with and without partial coverage.

Two experiments, both inference-only (no retraining):

  full   : every valid pair is decoded by (a) sigmoid(logit) > tau, and by
           (b) the transport plan (row-wise argmax, optional sink abstention).
  drop   : 10% of the B tracks are removed so that some A tracks have no
           counterpart; we then measure how often those orphan A tracks are
           still declared matched (false-link rate), which is exactly what the
           dustbin channel is supposed to prevent.

Usage::

    python -m tools.plan_decode --data_dir <scenes> --checkpoint seed_42.pt \
        --out plan_decode.json --max_scenes 300
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from phyglobt.dataset import cross_sensor_collate_fn
from phyglobt.evaluate import load_model, make_dataset

DELTAS = (0.4, 0.6)
RHOS = (0.2, 0.4, 0.5)


def _prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r + 1e-12), tp=tp, fp=fp, fn=fn)


def _counts(pred: torch.Tensor, rel: torch.Tensor, valid: torch.Tensor):
    pred = pred * valid
    y = rel * valid
    tp = float(((pred > 0.5) & (y > 0.5)).sum())
    fp = float(((pred > 0.5) & (y < 0.5)).sum())
    fn = float(((pred <= 0.5) & (y > 0.5)).sum())
    return tp, fp, fn


def _decode_plan(plan: torch.Tensor, n0: int, n1: int, delta: float,
                 col_mask: torch.Tensor | None = None, mass_min: float = 0.0) -> torch.Tensor:
    """Row-wise argmax over the transport plan; sink wins abstention."""
    pm = plan[:n0, :n1]
    # the plan is padded to the scene capacity, so the dustbin is the *last*
    # column of the extended matrix, not the first empty one
    sink = plan[:n0, -1]
    if col_mask is not None:
        pm = pm.masked_fill(~col_mask.unsqueeze(0), -1.0)
    idx = pm.argmax(dim=1)
    val = pm.gather(1, idx.unsqueeze(1)).squeeze(1)
    row_sum = pm.clamp(min=0).sum(dim=1) + 1e-12
    # under unbalanced OT a row may ship (almost) no mass, which is the
    # per-track abstention signal; balanced OT keeps this ratio near 1
    shipped = row_sum * n0
    keep = (val / row_sum > delta) & (val > sink) & (shipped > mass_min)
    pred = torch.zeros(n0, n1, device=pm.device)
    rows = torch.arange(n0, device=pm.device)[keep]
    pred[rows, idx[keep]] = 1.0
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_scenes', type=int, default=300)
    ap.add_argument('--drop_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', default='all', choices=['all', 'val'],
                    help="'val' = the 20% hold-out split used during training")
    ap.add_argument('--kappa', type=float, default=None,
                    help="override the KL-relaxation exponent (<1 = unbalanced OT)")
    ap.add_argument('--dump_scene', action='store_true',
                    help="store per-scene tp/fp/fn and the McNemar table for "
                         "paired significance tests")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt, cfg = load_model(args.checkpoint, device)
    if args.kappa is not None:
        model.sinkhorn.unbalanced_kappa = float(args.kappa)
    tau = float(ckpt.get('best_threshold', 0.5))
    ds = make_dataset(args.data_dir, cfg.get('max_trajectories', 256),
                      cfg.get('time_steps', 32), feature_mode=cfg.get('feature_mode', 'legacy'),
                      split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=cross_sensor_collate_fn)
    g = torch.Generator().manual_seed(args.seed)

    acc = {}
    def add(key, tp, fp, fn):
        a = acc.setdefault(key, [0.0, 0.0, 0.0])
        a[0] += tp; a[1] += fp; a[2] += fn

    orphan = {k: [0.0, 0.0] for k in [f'plan_rho{r}_d{d}' for r in RHOS for d in DELTAS]
              + ['threshold']}
    per_scene = []
    mcnemar = {f'plan_rho{r}_d{d}': [0, 0] for r in RHOS for d in DELTAS}
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            if n_done >= args.max_scenes:
                break
            traj_a, traj_b, rel, _, ts_a, ts_b, ma, mb, _ = batch
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            traj_a, traj_b = traj_a.to(device), traj_b.to(device)
            ts_a, ts_b, ma, mb = ts_a.to(device), ts_b.to(device), ma.to(device), mb.to(device)
            _, logits, _ = model(traj_a, traj_b, ts_a, ts_b, ma, mb)
            rel = rel.to(device)
            n0, n1 = int(ma.sum()), int(mb.sum())
            valid = (ma[0, :n0].bool().unsqueeze(-1) & mb[0, :n1].bool().unsqueeze(0))
            rel_s = rel[0, :n0, :n1]

            # (a) per-pair threshold, the rule used in the manuscript
            pred_thr = (torch.sigmoid(logits)[0, :n0, :n1] > tau).float()
            add('threshold', *_counts(pred_thr, rel_s, valid))
            scene = {'threshold': list(_counts(pred_thr, rel_s, valid))}
            if args.dump_scene:
                thr_hit = ((pred_thr > 0.5) == (rel_s > 0.5)) * valid

            # (b) plan decoding under several sink masses
            for rho in RHOS:
                plan = model.sinkhorn(logits, ma, mb, sink_mass=rho)[0]
                for d in DELTAS:
                    pred = _decode_plan(plan, n0, n1, d)
                    k = f'plan_rho{rho}_d{d}'
                    add(k, *_counts(pred, rel_s, valid))
                    if args.dump_scene:
                        scene[k] = list(_counts(pred, rel_s, valid))
                        plan_hit = ((pred > 0.5) == (rel_s > 0.5)) * valid
                        mcnemar[k][0] += int((~thr_hit.bool() & plan_hit.bool()).sum())
                        mcnemar[k][1] += int((thr_hit.bool() & ~plan_hit.bool()).sum())
            if args.dump_scene:
                per_scene.append(scene)

            # (c) partial coverage: remove a fraction of the B tracks
            keep_b = (torch.rand(n1, generator=g) >= args.drop_ratio).to(device)
            keep_full = torch.zeros(mb.shape[1], device=device)
            keep_full[:n1] = keep_b.float()
            if keep_b.sum() < 2:
                n_done += 1
                continue
            mb2 = mb.clone()
            mb2[0, :] = mb[0, :] * keep_full
            rel2 = rel.clone()
            rel2[:, :, :] = rel[:, :, :] * keep_full.view(1, 1, -1)
            _, logits2, _ = model(traj_a, traj_b, ts_a, ts_b, ma, mb2)
            # A tracks that lost their partner
            orphan_rows = (rel[0, :n0, :n1] * (1.0 - keep_full[:n1]).view(1, -1)).sum(dim=1) > 0
            n_orphan = int(orphan_rows.sum())
            if n_orphan > 0:
                pred_thr2 = (torch.sigmoid(logits2)[0, :n0, :n1] > tau).float()
                hit = (pred_thr2[orphan_rows].sum(dim=1) > 0).float().sum()
                orphan['threshold'][0] += float(hit); orphan['threshold'][1] += n_orphan
                for rho in RHOS:
                    plan2 = model.sinkhorn(logits2, ma, mb2, sink_mass=rho)[0]
                    for d in DELTAS:
                        pred2 = _decode_plan(plan2, n0, n1, d, col_mask=keep_b)
                        hit = (pred2[orphan_rows].sum(dim=1) > 0).float().sum()
                        k = f'plan_rho{rho}_d{d}'
                        orphan[k][0] += float(hit); orphan[k][1] += n_orphan
            n_done += 1

    report = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'scenes': n_done, 'threshold': tau, 'drop_ratio': args.drop_ratio,
        'full': {k: _prf(*v) for k, v in acc.items()},
        'orphan_false_link': {k: (v[0] / max(v[1], 1)) for k, v in orphan.items()},
        'orphan_counts': {k: v for k, v in orphan.items()},
    }
    if args.dump_scene:
        report['per_scene'] = per_scene
        report['mcnemar'] = mcnemar
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: round(v['f1'], 4) for k, v in report['full'].items()}, indent=1))
    print(json.dumps({k: round(v, 4) for k, v in report['orphan_false_link'].items()}, indent=1))


if __name__ == '__main__':
    main()
