# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""Novel-view metrics of trained runs on the held-out test images, in the Asteroid-GS
paper's terms (PSNR, SSIM, LPIPS; full frame, 8-bit range), two ways:

  raw      the renders as written by each method's render script
  matched  after one gain and offset per image, fitted by least squares over the foreground
           (observed mask or rendered body) and applied there. Methods with a per-image
           appearance code (Asteroid-GS, GOF: --use_decoupled_appearance) have no code for a
           held-out image and render it at a canonical exposure; matching removes that for
           every method alike, so the comparison is about shape and texture.

Writes <run>/nvs_table.json; skips runs that have one or have no renders. Foreground masks are read
from <run>/test_masks/00000.png..., which are copied from the training scene (source_path in
<run>/cfg_args, masks/ in sorted test-name order) when missing; else the image above 2 DN is used.
usage: python evaluation/nvs_metrics.py --run <model_dir> [...] [--threads 8] [--device cuda]
       (LPIPS with the VGG backbone)
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def gray(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    return im.astype(np.float64) / (65535.0 if im.dtype == np.uint16 else 255.0)


def metrics(gt, rd, fg, loss_fn):
    import torch
    from skimage.metrics import structural_similarity
    dev = next(loss_fn.parameters()).device
    t = lambda x: torch.from_numpy(x).float().to(dev)[None, None].repeat(1, 3, 1, 1) * 2 - 1  # noqa: E731
    with torch.no_grad():
        lp = float(loss_fn(t(gt), t(rd)).item())
    return dict(psnr=float(10 * np.log10(1.0 / max(((gt - rd) ** 2).mean(), 1e-12))),
                ssim=float(structural_similarity(gt, rd, data_range=1.0)), lpips=lp,
                fg_psnr=float(10 * np.log10(1.0 / max(((gt - rd)[fg] ** 2).mean(), 1e-12))))


def ensure_test_masks(M):
    """Copy the lit masks of the held-out images to <run>/test_masks/ in the order of the renders."""
    import re
    import shutil
    out = os.path.join(M, "test_masks")
    cfg = os.path.join(M, "cfg_args")
    if os.path.isdir(out) or not os.path.exists(cfg):
        return
    m = re.search(r"source_path='([^']+)'", open(cfg).read())
    lst = m and os.path.join(m.group(1), "test_images.txt")
    if not lst or not os.path.exists(lst):
        return
    names = sorted(l.strip() for l in open(lst) if l.strip())
    os.makedirs(out, exist_ok=True)
    for i, n in enumerate(names):
        shutil.copy(os.path.join(m.group(1), "masks", n), os.path.join(out, "%05d.png" % i))


def run_dir(M, loss_fn):
    ensure_test_masks(M)
    tests = sorted(glob.glob(f"{M}/test/ours_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
    if not tests:
        return None
    T = tests[-1]
    # renders/ + gt/ (3DGS, 2DGS, PGSR, Asteroid-GS, PCGS) or test_preds_*/ + gt_*/ (GOF)
    pick = lambda c: next((d for d in c if glob.glob(f"{d}/*.png")), None)  # noqa: E731
    rdir = pick([f"{T}/renders"] + sorted(glob.glob(f"{T}/test_preds_*")))
    gdir = pick([f"{T}/gt"] + sorted(glob.glob(f"{T}/gt_*")))
    if rdir is None or gdir is None:
        return None
    rows = []
    for gp in sorted(glob.glob(f"{gdir}/*.png")):
        stem = os.path.splitext(os.path.basename(gp))[0]
        rp = f"{rdir}/{stem}.png"
        if not os.path.exists(rp):
            continue
        gt, rd = gray(gp), gray(rp)
        mp = f"{M}/test_masks/{stem}.png"
        fg = (gray(mp) > 0.5) if os.path.exists(mp) else (gt > 2.0 / 255)
        fg = fg | (rd > 2.0 / 255)
        A = np.stack([rd[fg], np.ones(fg.sum())], 1)
        a, b = np.linalg.lstsq(A, gt[fg], rcond=None)[0]
        rm = rd.copy()
        rm[fg] = np.clip(a * rd[fg] + b, 0, 1)
        rows.append(dict(image=stem, gain=float(a), offset=float(b), raw=metrics(gt, rd, fg, loss_fn),
                         matched=metrics(gt, rm, fg, loss_fn)))
    mean = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in rows[0][v]} for v in ("raw", "matched")}
    return dict(renders=rdir, n=len(rows), mean=mean, per_image=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (LPIPS is ~10x faster on a GPU)")
    ap.add_argument("--run", nargs="+", required=True, help="run (model) directories")
    a = ap.parse_args()
    import torch
    import lpips
    torch.set_num_threads(a.threads)
    loss_fn = lpips.LPIPS(net="vgg", verbose=False).to(a.device)
    todo = [os.path.abspath(r) for r in a.run]
    for M in todo:
        tag = "/".join(M.replace(os.sep, "/").split("/")[-2:])     # <campaign>/<run>
        out = f"{M}/nvs_table.json"
        if os.path.exists(out) or not os.path.isdir(M):
            continue
        res = run_dir(M, loss_fn)
        if res is None:
            print("no renders", tag, flush=True)
            continue
        json.dump(res, open(out, "w"), indent=1)
        m = res["mean"]
        print(f"{tag}: raw PSNR {m['raw']['psnr']:.2f} / matched {m['matched']['psnr']:.2f} dB, "
              f"SSIM {m['matched']['ssim']:.4f}, LPIPS {m['matched']['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
