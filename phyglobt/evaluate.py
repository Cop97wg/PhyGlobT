"""Deterministic evaluation and analysis utilities for PhyGlobT.

Provides
--------
1. ``eval``      : evaluate one checkpoint over a scene directory, with per-scene
                   metrics, density-bin breakdown, and optional attention entropy.
2. ``compare``   : paired comparison (paired t-test / Wilcoxon) between two
                   checkpoints on identical scenes.
3. ``kde``       : physical-feature separability analysis from *real* features
                   extracted from two checkpoints (replaces the noise surrogate).
4. ``density``   : F1 and attention entropy over track-count bins (N sweep).

All commands write JSON (and NPZ for ``kde``), so cloud runs can be aggregated
afterwards without re-running inference.

Usage::

    python -m phyglobt.evaluate eval --data_dir data/scenes --checkpoint ckpt.pt --out eval.json
    python -m phyglobt.evaluate compare --data_dir data/scenes --ckpt_a a.pt --ckpt_b b.pt --out cmp.json
    python -m phyglobt.evaluate kde --data_dir data/scenes --ckpt_a a.pt --ckpt_b ckpt_b.pt --out kde.npz
    python -m phyglobt.evaluate density --data_dir data/scenes --checkpoint ckpt.pt --out density.json
"""

from __future__ import annotations

import argparse
import json
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CrossSensorTrajectoryDataset, cross_sensor_collate_fn
from .model import PhysiTrackModel

DEFAULT_MODEL_CONFIG = {
    'input_dim': 5, 'hidden_dim': 128, 'phys_dim': 64, 'lstm_layers': 2, 'dropout': 0.2,
    'use_topology': True, 'topo_num_layers': 2, 'topo_nhead': 8,
    'topo_dim_feedforward': 256, 'sinkhorn_iters': 20, 'sinkhorn_tau': 0.5,
    'sinkhorn_epsilon': 1e-3, 'dustbin_cost': 0.0, 'learn_dustbin_cost': False,
    'auto_dustbin_mass': True, 'dustbin_mass_ratio': 0.2, 'sinkhorn_stop_thresh': 1e-5,
    'use_transformer_encoder': False,
    'feature_mode': 'legacy', 'motion_model': 'cv', 'unbalanced_kappa': 1.0,
}

DATA_DEFAULTS = dict(
    max_trajectories=256, time_steps=32, use_interpolation=False, sparsify=True,
    max_time_gap=1800, max_distance=10.0, mask_augment=False, random_seed=42,
)


def load_model(checkpoint, device, model_config=None):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg.update(ckpt.get('model_config', {}) or {})
    state = ckpt.get('model_state_dict', {})
    if 'model_config' not in ckpt:
        # older checkpoints do not record their config; infer the topology flag
        cfg['use_topology'] = any(k.startswith('topology_encoder.') for k in state)
    cfg.update(model_config or {})
    model = PhysiTrackModel(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt, cfg


def make_dataset(data_dir, max_trajectories=256, time_steps=32, seed=42,
                 feature_mode='legacy', split='all'):
    """Scene split used for evaluation.

    ``all``  : every scene in ``data_dir`` (use for a separate held-out corpus)
    ``val``  : the 20% validation split used by training (``train_ratio=0.8``),
               so evaluation never touches the scenes the model was trained on.
    """
    train_ratio, test_ratio = (0.8, 0.0) if split == 'val' else (0.0, 1.0)
    return CrossSensorTrajectoryDataset(
        data_folder=data_dir, mode='val' if split == 'val' else 'test',
        max_trajectories=max_trajectories, time_steps=time_steps,
        sparsify=True, mask_augment=False, random_seed=seed,
        train_ratio=train_ratio, test_ratio=test_ratio, lazy_load=True,
        feature_mode=feature_mode,
    )


def _attention_stats(model, fused, mask):
    """Mean attention entropy / off-diagonal mass of the Phy-GTE layers."""
    if not getattr(model, 'use_topology', False):
        return {}
    try:  # the MHA fast path uses NestedTensor internally and breaks the hooks
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    n_valid = int(mask.sum().item())
    if n_valid < 3:
        return {}

    captured = {}
    handles = []

    def make_hook(idx):
        def hook(mod, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get('query')
            captured[idx] = x.detach()
        return hook

    for idx, layer in enumerate(model.topology_encoder.transformer.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(idx), with_kwargs=True))
    with torch.no_grad():
        model.topology_encoder(fused, mask)
    for h in handles:
        h.remove()

    out = {}
    for idx, x in captured.items():
        attn_mod = model.topology_encoder.transformer.layers[idx].self_attn
        d = attn_mod.embed_dim
        heads = attn_mod.num_heads
        dh = d // heads
        w = attn_mod.in_proj_weight
        b = attn_mod.in_proj_bias
        # valid tracks are stored first, padding is appended at the end
        x = x[:, :n_valid]
        q = F.linear(x, w[:d], b[:d])
        k = F.linear(x, w[d:2 * d], b[d:2 * d])
        B, N, _ = q.shape
        q = q.view(B, N, heads, dh).transpose(1, 2)
        k = k.view(B, N, heads, dh).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(dh)
        p = torch.softmax(logits, dim=-1)
        ent = -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum(-1)  # (B, heads, N)
        eye = torch.eye(N, device=p.device).bool()
        off = p.masked_fill(eye[None, None], 0.0)
        out[f'layer{idx + 1}'] = {
            'entropy_mean': float(ent.mean().item()),
            'entropy_norm': float(ent.mean().item() / math.log(max(N - 1, 2))),
            'offdiag_mean': float(off.sum().item() / max(heads * N * (N - 1), 1)),
            'n_tracks': N,
        }
    return out


def _confusion(probs, relations, mask_a, mask_b, thresh):
    valid = mask_a.unsqueeze(-1) * mask_b.unsqueeze(1)
    pred = (probs > thresh).float() * valid
    y = relations * valid
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return tp, fp, fn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def run_eval(data_dir, checkpoint, device, threshold=None, bins=(50, 100, 200, 400),
             attention=False, max_scenes=None, batch_size=1, split='all', split_seed=None):
    model, ckpt, cfg = load_model(checkpoint, device)
    thresh = float(ckpt.get('best_threshold', 0.5)) if threshold is None else float(threshold)
    if split_seed is None:
        split_seed = int(ckpt.get('seed', 42))
    ds = make_dataset(data_dir, cfg.get('max_trajectories', 256), cfg.get('time_steps', 32),
                      seed=split_seed, feature_mode=cfg.get('feature_mode', 'legacy'),
                      split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=cross_sensor_collate_fn)

    rows = []
    agg = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
    bin_agg = {}
    attn_collect = {}
    attn_bin = {}
    n_done = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_scenes is not None and n_done >= max_scenes:
                break
            traj_a, traj_b, rel, _, ts_a, ts_b, ma, mb, _ = batch
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            traj_a, traj_b = traj_a.to(device), traj_b.to(device)
            ts_a, ts_b = ts_a.to(device), ts_b.to(device)
            ma, mb = ma.to(device), mb.to(device)
            _, logits, aux = model(traj_a, traj_b, ts_a, ts_b, ma, mb)
            probs = torch.sigmoid(logits)
            tp, fp, fn = _confusion(probs.cpu(), rel, ma.cpu(), mb.cpu(), thresh)
            n1, n2 = int(ma.sum()), int(mb.sum())
            name = os.path.basename(ds.scenario_files[i]) if i < len(ds.scenario_files) else f'scene_{i}'
            row = dict(scene=name, n_a=n1, n_b=n2, threshold=thresh)
            row.update(_prf(tp, fp, fn))
            rows.append(row)
            n_done += 1
            agg['tp'] += tp
            agg['fp'] += fp
            agg['fn'] += fn

            # density bin: label by max(n_a, n_b) so scenes with one-sided
            # coverage are not mis-binned
            n_max = max(n1, n2)
            label = _bin_label(n_max, bins)
            b = bin_agg.setdefault(label, {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'scenes': 0})
            b['tp'] += tp
            b['fp'] += fp
            b['fn'] += fn
            b['scenes'] += 1

            if attention:
                fused_a = torch.cat([aux[2], aux[4]], dim=-1)
                stats = _attention_stats(model, fused_a, ma)
                for k, v in stats.items():
                    for target in (attn_collect, attn_bin.setdefault(label, {})):
                        rec = target.setdefault(k, {'entropy_mean': 0.0, 'entropy_norm': 0.0,
                                                    'offdiag_mean': 0.0, 'scenes': 0, 'n_tracks': 0})
                        rec['entropy_mean'] += v['entropy_mean']
                        rec['entropy_norm'] += v['entropy_norm']
                        rec['offdiag_mean'] += v['offdiag_mean']
                        rec['scenes'] += 1
                        rec['n_tracks'] += v['n_tracks']

    per_bin = {k: {**_prf(v['tp'], v['fp'], v['fn']), 'scenes': v['scenes']}
               for k, v in bin_agg.items()}
    out = {
        'checkpoint': os.path.abspath(checkpoint),
        'data_dir': os.path.abspath(data_dir),
        'threshold': thresh,
        'n_scenes': n_done,
        'overall': _prf(agg['tp'], agg['fp'], agg['fn']),
        'by_density': per_bin,
        'per_scene': rows,
    }
    if attention:
        def _avg(d):
            return {k: {kk: (vv / v['scenes'] if kk in
                              ('entropy_mean', 'entropy_norm', 'offdiag_mean')
                              else vv) for kk, vv in v.items()}
                    for k, v in d.items()}
        out['attention'] = _avg(attn_collect)
        out['attention_by_density'] = {lab: _avg(d) for lab, d in attn_bin.items()}
    return out


def _bin_label(n, bins):
    edges = sorted(bins)
    if n < edges[0]:
        return f'<{edges[0]}'
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= n < hi:
            return f'{lo}-{hi}'
    return f'>={edges[-1]}'


def run_compare(data_dir, ckpt_a, ckpt_b, device, max_scenes=None, alpha=0.05):
    """Paired comparison of two checkpoints on identical scenes."""
    ea = run_eval(data_dir, ckpt_a, device, max_scenes=max_scenes, attention=False)
    eb = run_eval(data_dir, ckpt_b, device, max_scenes=max_scenes, attention=False)
    fa = {r['scene']: r['f1'] for r in ea['per_scene']}
    fb = {r['scene']: r['f1'] for r in eb['per_scene']}
    keys = sorted(set(fa) & set(fb))
    a = np.array([fa[k] for k in keys])
    b = np.array([fb[k] for k in keys])
    diff = a - b
    result = {
        'n_scenes': len(keys),
        'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'mean_diff': float(diff.mean()), 'std_diff': float(diff.std(ddof=1)) if len(keys) > 1 else 0.0,
    }
    if len(keys) >= 3:
        try:
            from scipy import stats
            t, p = stats.ttest_rel(a, b)
            result['paired_t'] = {'t': float(t), 'p': float(p)}
            if len(keys) >= 6:
                w, pw = stats.wilcoxon(a, b)
                result['wilcoxon'] = {'stat': float(w), 'p': float(pw)}
        except ImportError:
            result['note'] = 'scipy not installed; install scipy for paired tests'
    return result


def extract_pair_features(model, loader, device, max_scenes=None):
    """Collect (phys_A, phys_B, label, scene, n) for every valid pair."""
    feats_pos, feats_neg, meta = [], [], []
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if max_scenes is not None and n_done >= max_scenes:
                break
            traj_a, traj_b, rel, _, ts_a, ts_b, ma, mb, _ = batch
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            traj_a, traj_b = traj_a.to(device), traj_b.to(device)
            ts_a, ts_b = ts_a.to(device), ts_b.to(device)
            ma, mb = ma.to(device), mb.to(device)
            _, _, aux = model(traj_a, traj_b, ts_a, ts_b, ma, mb)
            pa, pb = aux[4].cpu(), aux[5].cpu()
            valid = (ma.cpu().bool().unsqueeze(-1) & mb.cpu().bool().unsqueeze(1))
            rel = rel.cpu()
            for i in range(rel.shape[0]):
                for j in range(rel.shape[1]):
                    if not valid[0, i, j]:
                        continue
                    d = torch.norm(pa[0, i] - pb[0, j]).item()
                    (feats_pos if rel[0, i, j] == 1 else feats_neg).append(d)
            meta.append(dict(n_a=int(ma.sum()), n_b=int(mb.sum())))
            n_done += 1
    return np.array(feats_pos), np.array(feats_neg), meta


def run_kde(data_dir, ckpt_a, ckpt_b, device, out_path, max_scenes=None):
    """Compare matched vs unmatched distance distributions for two checkpoints."""
    report = {}
    npz = {}
    for tag, path in (('model_a', ckpt_a), ('model_b', ckpt_b)):
        model, _, cfg = load_model(path, device)
        ds = make_dataset(data_dir, cfg.get('max_trajectories', 256), cfg.get('time_steps', 32),
                      feature_mode=cfg.get('feature_mode', 'legacy'))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=cross_sensor_collate_fn)
        pos, neg, meta = extract_pair_features(model, loader, device, max_scenes)
        sep = float(neg.mean() / max(pos.mean(), 1e-9)) if pos.size and neg.size else float('nan')
        report[tag] = {
            'checkpoint': os.path.abspath(path),
            'n_positive': int(pos.size), 'n_negative': int(neg.size),
            'pos_mean': float(pos.mean()) if pos.size else None,
            'neg_mean': float(neg.mean()) if neg.size else None,
            'pos_std': float(pos.std()) if pos.size else None,
            'neg_std': float(neg.std()) if neg.size else None,
            'separation_ratio': sep,
            'scenes': len(meta),
        }
        npz[f'{tag}_pos'] = pos
        npz[f'{tag}_neg'] = neg
    np.savez_compressed(out_path, **npz)
    with open(os.path.splitext(out_path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def parse_args():
    p = argparse.ArgumentParser(description='PhyGlobT evaluation / analysis')
    sub = p.add_subparsers(dest='cmd', required=True)

    e = sub.add_parser('eval')
    e.add_argument('--data_dir', required=True)
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--threshold', type=float, default=None)
    e.add_argument('--bins', type=str, default='50,100,200,400')
    e.add_argument('--attention', action='store_true')
    e.add_argument('--max_scenes', type=int, default=None)

    c = sub.add_parser('compare')
    c.add_argument('--data_dir', required=True)
    c.add_argument('--ckpt_a', required=True)
    c.add_argument('--ckpt_b', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--max_scenes', type=int, default=None)

    k = sub.add_parser('kde')
    k.add_argument('--data_dir', required=True)
    k.add_argument('--ckpt_a', required=True)
    k.add_argument('--ckpt_b', required=True)
    k.add_argument('--out', required=True, help='output .npz (JSON written alongside)')
    k.add_argument('--max_scenes', type=int, default=None)

    d = sub.add_parser('density')
    d.add_argument('--data_dir', required=True)
    d.add_argument('--checkpoint', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--bins', type=str, default='50,100,200,400')
    d.add_argument('--max_scenes', type=int, default=None)

    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.cmd == 'eval':
        res = run_eval(args.data_dir, args.checkpoint, device, args.threshold,
                       tuple(int(x) for x in args.bins.split(',')), args.attention,
                       args.max_scenes)
        _dump(res, args.out)
    elif args.cmd == 'density':
        res = run_eval(args.data_dir, args.checkpoint, device, None,
                       tuple(int(x) for x in args.bins.split(',')), True, args.max_scenes)
        _dump({'checkpoint': res['checkpoint'], 'n_scenes': res['n_scenes'],
               'overall': res['overall'], 'by_density': res['by_density'],
               'attention': res.get('attention', {}),
               'attention_by_density': res.get('attention_by_density', {})},
              args.out)
    elif args.cmd == 'compare':
        _dump(run_compare(args.data_dir, args.ckpt_a, args.ckpt_b, device, args.max_scenes),
              args.out)
    elif args.cmd == 'kde':
        _dump(run_kde(args.data_dir, args.ckpt_a, args.ckpt_b, device, args.out,
                      args.max_scenes), args.out.replace('.npz', '.json'))


def _dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f'written: {path}')


if __name__ == '__main__':
    main()
