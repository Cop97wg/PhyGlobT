"""Paired significance analysis for the plan-decoding experiments.

Reads one ``plan_decode --dump_scene`` report per checkpoint and reports, for
every decoding rule,

  * the F1 with a 95% bootstrap confidence interval (scenes resampled with
    replacement, which respects the scene-level dependence of the pairs),
  * a paired Wilcoxon signed-rank test of the per-scene F1 against the
    per-pair threshold rule,
  * a McNemar test on the pair-level decisions, using the disagreement table
    accumulated while decoding.

Usage::

    python -m tools.significance --input eval/ext1000_k0.5.json --out sig.json
    python -m tools.significance --input a.json b.json c.json --out sig.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats


def _f1(tp, fp, fn):
    tp, fp, fn = np.asarray(tp), np.asarray(fp), np.asarray(fn)
    return 2 * tp / (2 * tp + fp + fn + 1e-12)


def _bootstrap(tp, fp, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(tp)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1 = _f1(tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1))
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def _mcnemar(b, c):
    if b + c == 0:
        return 1.0, 0.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stats.chi2.sf(chi2, 1)), float(chi2)


def analyse(path):
    d = json.load(open(path, encoding='utf-8'))
    scenes = d['per_scene']
    keys = [k for k in scenes[0]]
    out = {'checkpoint': d['checkpoint'], 'scenes': len(scenes), 'rules': {}}
    ref = 'threshold'
    ref_tp = np.array([s[ref][0] for s in scenes])
    ref_fp = np.array([s[ref][1] for s in scenes])
    ref_fn = np.array([s[ref][2] for s in scenes])
    ref_f1 = _f1(ref_tp, ref_fp, ref_fn)
    ref_pooled = float(_f1(ref_tp.sum(), ref_fp.sum(), ref_fn.sum()))
    # scenes without a single positive pair carry no information for the test
    informative = (ref_tp + ref_fn) > 0

    for k in keys:
        tp = np.array([s[k][0] for s in scenes])
        fp = np.array([s[k][1] for s in scenes])
        fn = np.array([s[k][2] for s in scenes])
        f1 = _f1(tp, fp, fn)
        lo, hi = _bootstrap(tp, fp, fn)
        entry = {
            'tp': float(tp.sum()), 'fp': float(fp.sum()), 'fn': float(fn.sum()),
            'precision': float(tp.sum() / (tp.sum() + fp.sum() + 1e-12)),
            'recall': float(tp.sum() / (tp.sum() + fn.sum() + 1e-12)),
            'f1': float(_f1(tp.sum(), fp.sum(), fn.sum())),
            'f1_ci95': [lo, hi],
            'scenes_improved': int((f1 > ref_f1 + 1e-9).sum()),
            'scenes_worsened': int((f1 < ref_f1 - 1e-9).sum()),
        }
        if k != ref:
            diff = f1 - ref_f1
            if np.allclose(diff, 0):
                entry['wilcoxon_p'] = 1.0
            else:
                entry['wilcoxon_p'] = float(stats.wilcoxon(f1[informative],
                                                          ref_f1[informative]).pvalue)
            b, c = d.get('mcnemar', {}).get(k, [0, 0])
            p_mc, chi2 = _mcnemar(b, c)
            entry['mcnemar'] = {'plan_right_thr_wrong': int(b),
                                'thr_right_plan_wrong': int(c),
                                'chi2': chi2, 'p': p_mc}
            entry['f1_delta'] = float(entry['f1'] - ref_pooled)
        out['rules'][k] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    runs = [analyse(p) for p in args.input]
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(runs, f, indent=2)

    for r in runs:
        print(f"\n== {os.path.basename(r['checkpoint'])}  ({r['scenes']} scenes)")
        print(f"{'rule':18s} {'F1':>7s} {'95% CI':>16s} {'dF1':>7s} {'Wilcoxon p':>11s} {'McNemar p':>10s}")
        for k, e in r['rules'].items():
            ci = f"[{e['f1_ci95'][0]:.4f},{e['f1_ci95'][1]:.4f}]"
            p = e.get('wilcoxon_p')
            pm = e.get('mcnemar', {}).get('p')
            print(f"{k:18s} {e['f1']:7.4f} {ci:>16s} "
                  f"{e.get('f1_delta', 0.0):+7.4f} "
                  f"{('%.2e' % p) if p is not None else '--':>11s} "
                  f"{('%.1e' % pm) if pm is not None else '--':>10s}")


if __name__ == '__main__':
    main()
