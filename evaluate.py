"""Comparative evaluation + ablation  ->  eval_P1_<TEAM>.md

Examples
  # real paired data (LOL eval15 / LOL-v2): <dir>/low/*.png and <dir>/high/*.png
  python evaluate.py --data_dir data/eval15 --team T01 --usns "1AB21CS001,1AB21CS002"
  # no ground truth (unseen images): no-reference metrics only
  python evaluate.py --noref_dir samples --team T01
  # self-contained smoke test on synthetic pairs (NOT a substitute for LOL numbers)
  python evaluate.py --synthetic --team T01
"""
import argparse
import glob
import json
import os
import time
from dataclasses import fields

import cv2
import numpy as np

from lowlight.pipeline import LowLightPipeline, Toggles
from lowlight.metrics import evaluate_all, nr_quality
from lowlight.classical import msrcr

DOC_ORDER = ['PSNR', 'SSIM', 'LPIPS', 'NIQE', 'NatProxy', 'Latency_ms', 'NoiseRatio', 'EdgePres', 'Overexp%',
             'HueShift', 'DeltaE00', 'Detail']


def load_img(path, max_side):
    im = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if max_side and max(im.shape[:2]) > max_side:
        s = max_side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


def load_pairs(data_dir, max_side):
    lows = sorted(glob.glob(os.path.join(data_dir, 'low', '*.*')))
    pairs = []
    for lp in lows:
        hp = os.path.join(data_dir, 'high', os.path.basename(lp))
        if not os.path.exists(hp):
            cands = glob.glob(os.path.join(data_dir, 'high', os.path.splitext(os.path.basename(lp))[0] + '.*'))
            if not cands:
                continue
            hp = cands[0]
        lo, hi = load_img(lp, max_side), load_img(hp, max_side)
        if lo.shape == hi.shape:
            pairs.append((os.path.basename(lp), lo, hi))
    return pairs


def synthetic_pairs(max_side):
    from lowlight.synth import natural_test_images, make_low_light
    out = []
    for n, gt in natural_test_images(max_side).items():
        for lvl in (0.04, 0.08, 0.15):
            out.append((f'{n}_L{lvl}', make_low_light(gt, level=lvl, seed=3), gt))
    return out


def fmt(v, nd=3):
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{nd}f}'


def mean_metrics(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def table(title_row, data, cols):
    lines = ['| ' + ' | '.join([title_row] + cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for name, m in data.items():
        lines.append('| ' + ' | '.join([name] + [fmt(m.get(c), 2 if c in ('PSNR', 'Latency_ms', 'Overexp%', 'HueShift', 'DeltaE00') else 3) for c in cols]) + ' |')
    return '\n'.join(lines)


def run_method(pipe, name, low, gt):
    t0 = time.perf_counter()
    if name == 'Classical (MSRCR baseline)':
        out = msrcr(low)
    elif name == 'Classical (ours)':
        out = pipe.process(low, 'classical').output
    elif name == 'Deep learning (RetinexNet)':
        out = pipe.process(low, 'dl').output
    else:
        out = pipe.process(low, 'hybrid').output
    dt = (time.perf_counter() - t0) * 1000
    m = evaluate_all(out, low, gt)
    m['Latency_ms'] = dt
    return out, m


def save_grid(path, imgs):
    h = min(i.shape[0] for i in imgs)
    row = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
    cv2.imwrite(path, cv2.cvtColor((np.concatenate(row, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir')
    ap.add_argument('--noref_dir')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--ckpt_dir', default='ckpts')
    ap.add_argument('--team', default='T01')
    ap.add_argument('--usns', default='USN1,USN2,USN3,USN4')
    ap.add_argument('--max_side', type=int, default=400)
    ap.add_argument('--out_dir', default='results')
    ap.add_argument('--no_ablation', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='', help='suffix for the output file name, e.g. _synthetic')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, 'figures'), exist_ok=True)

    pipe = LowLightPipeline(a.ckpt_dir, threads=None)
    has_gt = True
    if a.data_dir:
        pairs = load_pairs(a.data_dir, a.max_side)
        src = f'paired dataset `{a.data_dir}`'
    elif a.noref_dir:
        files = sorted(glob.glob(os.path.join(a.noref_dir, '*.*')))
        pairs = [(os.path.basename(f), load_img(f, a.max_side), None) for f in files]
        has_gt, src = False, f'unseen images without ground truth `{a.noref_dir}`'
    else:
        pairs = synthetic_pairs(a.max_side)
        src = 'SYNTHETIC low-light pairs (skimage photos + physically-motivated degradation; smoke test only)'
    if a.limit:
        pairs = pairs[:a.limit]
    assert pairs, 'no images found'
    print(f'{len(pairs)} images | {src}')

    methods = ['Classical (MSRCR baseline)', 'Classical (ours)']
    if pipe.dl_available:
        methods += ['Deep learning (RetinexNet)', 'Hybrid (ours)']
    per = {m: [] for m in methods}
    per_img = {}
    # warm-up (first call pays import / allocation costs)
    run_method(pipe, methods[-1], pairs[0][1], pairs[0][2])
    for name, low, gt in pairs:
        outs = {'Input': low}
        for m in methods:
            o, mm = run_method(pipe, m, low, gt)
            per[m].append(mm)
            outs[m] = o
            per_img.setdefault(name, {})[m] = mm
        if len(per_img) <= 12:
            stem = os.path.splitext(name)[0] if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) else name
            save_grid(os.path.join(a.out_dir, 'figures', f'compare{a.tag}_{stem.replace(".", "p")}.jpg'),
                      list(outs.values()) + ([gt] if gt is not None else []))
    summary = {m: mean_metrics(r) for m, r in per.items()}

    # ---------------------------------------------------------------- ablation (hybrid only)
    abl = {}
    if pipe.dl_available and not a.no_ablation:
        pipe.dl.cache = True
        variants = {'Full hybrid': {}}
        for f in fields(Toggles):
            variants[f'− {f.name}'] = {f.name: False}
        variants['Classical only (no DL)'] = None
        variants['DL only (raw RetinexNet)'] = 'dl'
        for vn, tgl in variants.items():
            rows = []
            for name, low, gt in pairs:
                if tgl is None:
                    out = pipe.process(low, 'classical').output
                elif tgl == 'dl':
                    out = pipe.process(low, 'dl').output
                else:
                    out = pipe.process(low, 'hybrid', toggles=Toggles(**tgl)).output
                rows.append(evaluate_all(out, low, gt, with_slow=False))
            abl[vn] = mean_metrics(rows)
            print('ablation', vn, {k: fmt(abl[vn].get(k)) for k in ('PSNR', 'SSIM', 'NoiseRatio')})
        pipe.dl.cache = False

    # ---------------------------------------------------------------- failure candidates
    fails = []
    if pipe.dl_available:
        for name, m in per_img.items():
            h = m['Hybrid (ours)']
            fails.append((name, h.get('PSNR'), h['Overexp%'], h['HueShift'], h['NoiseRatio'], h['EdgePres']))
        key = (lambda x: x[1]) if has_gt else (lambda x: -x[5])
        fails.sort(key=key)

    # ---------------------------------------------------------------- write markdown
    cols = [c for c in DOC_ORDER if any(summary[m].get(c) is not None for m in summary)]
    if not has_gt:
        cols = [c for c in cols if c not in ('PSNR', 'SSIM', 'LPIPS', 'DeltaE00')]
    md = [a.usns, '',
          f'# Evaluation - ProbID P1 - Team {a.team}', '',
          f'**Data:** {src} - {len(pairs)} images, max side {a.max_side}px, CPU inference.', '',
          '> `NIQE` is the official metric only when `pyiqa` is installed (otherwise n/a); `NatProxy` is our uncalibrated '
          'NIQE-style MSCN naturalness proxy (lower = more natural) and must NOT be reported as NIQE. `LPIPS` requires the '
          '`lpips` package and internet access for backbone weights (n/a otherwise).',
          '> `NoiseRatio` = estimated noise sigma / mean luma (lower = cleaner). `EdgePres` = gradient-map correlation with the '
          'reference (GT if available, else input; higher = better). `Overexp%` = newly clipped pixels. `HueShift` = '
          'chroma-weighted LAB hue change in degrees (lower = fewer colour shifts).', '',
          '## 1. Approach comparison (classical vs deep learning vs hybrid)', '',
          table('Method', summary, cols), '']
    if abl:
        acols = [c for c in ['PSNR', 'SSIM', 'DeltaE00', 'NoiseRatio', 'EdgePres', 'Overexp%', 'HueShift', 'MeanLuma']
                 if abl['Full hybrid'].get(c) is not None]
        md += ['## 2. Ablation (each row removes one component from the full hybrid)', '', table('Variant', abl, acols), '']
    if fails:
        md += ['## 3. Lowest-scoring images for the hybrid (failure-analysis candidates)', '',
               '| Image | PSNR | Overexp% | HueShift | NoiseRatio | EdgePres |', '|---|---|---|---|---|---|']
        for f in fails[:5]:
            md.append(f'| {f[0]} | {fmt(f[1], 2)} | {fmt(f[2], 2)} | {fmt(f[3], 1)} | {fmt(f[4])} | {fmt(f[5])} |')
        md += ['', f'Side-by-side figures (Input | MSRCR | Classical | DL | Hybrid{" | GT" if has_gt else ""}) are in `{a.out_dir}/figures/`.', '']
    path = os.path.join(a.out_dir, f'eval_P1_{a.team}{a.tag}.md')
    open(path, 'w').write('\n'.join(md))
    json.dump({'summary': summary, 'ablation': abl, 'per_image': per_img}, open(os.path.join(a.out_dir, f'eval_raw{a.tag}.json'), 'w'),
              indent=1, default=float)
    print('wrote', path)


if __name__ == '__main__':
    main()
