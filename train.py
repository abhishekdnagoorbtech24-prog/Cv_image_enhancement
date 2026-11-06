"""Train / fine-tune RetinexNet (CPU or GPU).  Replaces the uploaded train.py + RetinexNet.train().

Fixes vs. the original code
  * no hard-coded `.cuda()`; `--device auto|cpu|cuda`
  * real shuffling (original `random.shuffle(list(tmp))` shuffled a throw-away copy => never shuffled)
  * torch DataLoader (multi-worker, no per-patch file re-reads inside the training loop)
  * explicit train/validation split + validation PSNR logged and best checkpoint kept
  * DecomNet frozen during the Relight phase (its gradients were computed but never applied)
  * `--init_ckpt` fine-tuning from the pretrained 9200-step weights (default) and optional
    `--noise_aug` that adds Poisson-Gaussian-like noise to LOW patches so the network adapts to sensor noise

Data layout: <data_dir>/low/*.png  <data_dir>/high/*.png   (same file names; LOL, LOL-v2 real/synthetic all fit)
Usage:
  python train.py --data_dir data/LOL/our485 --val_dir data/LOL/eval15 --epochs 30 --noise_aug 0.02
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lowlight.retinexnet import RetinexNet, retinex_losses, latest_ckpt, save_ckpt


class PairDataset(Dataset):
    def __init__(self, pairs, patch, noise_aug=0.0, train=True):
        self.pairs, self.patch, self.noise, self.train = pairs, patch, noise_aug, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lo = np.asarray(Image.open(self.pairs[i][0]).convert('RGB'), np.float32) / 255.0
        hi = np.asarray(Image.open(self.pairs[i][1]).convert('RGB'), np.float32) / 255.0
        h, w, _ = lo.shape
        p = min(self.patch, h, w)
        y, x = random.randint(0, h - p), random.randint(0, w - p)
        lo, hi = lo[y:y + p, x:x + p], hi[y:y + p, x:x + p]
        if self.train:
            if random.random() < 0.5: lo, hi = lo[::-1], hi[::-1]
            if random.random() < 0.5: lo, hi = lo[:, ::-1], hi[:, ::-1]
            k = random.randint(0, 3)
            lo, hi = np.rot90(lo, k), np.rot90(hi, k)
            if self.noise > 0:                       # signal-dependent + read noise on the dark input only
                s = random.uniform(0, self.noise)
                lo = np.clip(lo + np.random.randn(*lo.shape).astype(np.float32) * s * (0.3 + np.sqrt(lo)), 0, 1)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return f(lo), f(hi)


def find_pairs(d):
    out = []
    for lp in sorted(glob.glob(os.path.join(d, 'low', '*.*'))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        hp = glob.glob(os.path.join(d, 'high', stem + '.*'))
        if hp:
            out.append((lp, hp[0]))
    return out


@torch.no_grad()
def validate(net, pairs, device, max_side=400):
    net.eval()
    ps = []
    for lp, hp in pairs:
        lo = Image.open(lp).convert('RGB'); hi = Image.open(hp).convert('RGB')
        if max(lo.size) > max_side:
            s = max_side / max(lo.size)
            sz = (int(lo.size[0] * s), int(lo.size[1] * s))
            lo, hi = lo.resize(sz, Image.BILINEAR), hi.resize(sz, Image.BILINEAR)
        lo = torch.from_numpy(np.asarray(lo, np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
        hi = torch.from_numpy(np.asarray(hi, np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
        R, I = net.DecomNet(lo)
        S = (R * net.RelightNet(I, R)).clamp(0, 1)
        mse = torch.mean((S - hi) ** 2).item()
        ps.append(10 * np.log10(1.0 / max(mse, 1e-10)))
    net.train()
    return float(np.mean(ps)) if ps else float('nan')


def run_phase(net, phase, loader, val_pairs, args, device, step0):
    params = net.DecomNet.parameters() if phase == 'Decom' else net.RelightNet.parameters()
    for p_ in net.DecomNet.parameters():
        p_.requires_grad_(phase == 'Decom')
    for p_ in net.RelightNet.parameters():
        p_.requires_grad_(phase == 'Relight')
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    step, best = step0, -1
    t0 = time.time()
    for ep in range(args.epochs):
        lr = args.lr if ep < args.lr_drop_epoch else args.lr / 10
        for g in opt.param_groups:
            g['lr'] = lr
        tot, n = 0.0, 0
        for lo, hi in loader:
            lo, hi = lo.to(device), hi.to(device)
            ld, lr_ = retinex_losses(net, lo, hi)
            loss = ld if phase == 'Decom' else lr_
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1; step += 1
        msg = f'[{phase}] epoch {ep + 1}/{args.epochs} loss {tot / max(n, 1):.5f} lr {lr:g} ({time.time() - t0:.0f}s)'
        if phase == 'Relight' and val_pairs and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            psnr = validate(net, val_pairs, device)
            msg += f' | val PSNR {psnr:.2f} dB'
            if psnr > best:
                best = psnr
                save_ckpt(net, args.ckpt_dir, 'Relight', step)
                save_ckpt(net, args.ckpt_dir, 'Decom', step)
                msg += '  (saved)'
        print(msg, flush=True)
    if phase == 'Decom' or not val_pairs:
        save_ckpt(net, args.ckpt_dir, phase, step)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--val_dir', default=None, help='paired validation folder; default: hold out --val_split of train')
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--ckpt_dir', default='ckpts_finetuned')
    ap.add_argument('--init_ckpt', default='ckpts', help="pretrained weights to start from ('' = from scratch)")
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--patch_size', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1e-4, help='fine-tuning LR (use 1e-3 when training from scratch)')
    ap.add_argument('--lr_drop_epoch', type=int, default=20)
    ap.add_argument('--noise_aug', type=float, default=0.0, help='max noise sigma added to low patches (0 = off)')
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    pairs = find_pairs(args.data_dir)
    assert pairs, f'no low/high pairs under {args.data_dir}'
    if args.val_dir:
        val = find_pairs(args.val_dir)
    else:
        random.Random(args.seed).shuffle(pairs)
        k = max(1, int(len(pairs) * args.val_split))
        val, pairs = pairs[:k], pairs[k:]
    print(f'{len(pairs)} train / {len(val)} val pairs | device {device}')
    net = RetinexNet().to(device).train()
    if args.init_ckpt:
        for ph, m in (('Decom', net.DecomNet), ('Relight', net.RelightNet)):
            c = latest_ckpt(args.init_ckpt, ph)
            if c:
                m.load_state_dict(torch.load(c, map_location=device, weights_only=True))
                print('loaded', c)
    loader = DataLoader(PairDataset(pairs, args.patch_size, args.noise_aug), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=len(pairs) >= args.batch_size)
    step = run_phase(net, 'Decom', loader, val, args, device, 0)
    run_phase(net, 'Relight', loader, val, args, device, step)
    print('done ->', args.ckpt_dir)


if __name__ == '__main__':
    main()
