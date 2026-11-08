"""Fig. 4 (Nature Communications layout, 183 mm): photometric consistency on held-out images.

Tiled on black like Fig. 2. Columns: a held-out Ryugu image and the renders of PCGS, PGSR and
Asteroid-GS for its camera after the per-image gain and offset of the novel-view table
(evaluation/nvs_metrics.py, 'matched').
  a  full views (zoom window marked)       b  the zoom window
  c  per-image PSNR of every held-out image (all splits), Itokawa and Ryugu
  d  residual render - image on the body
usage: python nc_fig_nvs.py <dir with renders and nvs tables> <out>
"""
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402,F401
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import result_figs as rf  # noqa: E402

mm = 1 / 25.4
BODY, NAME = "ryugu", "hyb2_onc_20180710_060508_tvf_l2c"
METHODS = [("pcgs-v18n", "PCGS (this work)"), ("pgsr", "PGSR"), ("ags", "Asteroid-GS")]
RMAX = 0.15
ZOOM = rf.ZOOM


def gray(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    return im.astype(np.float64) / 255.0


def per_image(d, body, tag, split=0):
    f = os.path.join(d, f"{body}__{tag}__s{split}.json")
    return json.load(open(f))["per_image"] if os.path.exists(f) else []


def main():
    d, out = sys.argv[1:3]
    rf.style()
    gt = gray(os.path.join(d, f"{BODY}_pcgs-v18n_gt.png"))
    fg = gray(os.path.join(d, f"{BODY}_pcgs-v18n_mask.png")) > 0.5
    rend, stats = {}, {}
    for tag, _ in METHODS:
        rd = gray(os.path.join(d, f"{BODY}_{tag}_render.png"))
        row = next(r for r in per_image(d, BODY, tag) if r["image"] in ("00000", NAME))
        m = fg | (rd > 2.0 / 255)
        rm = rd.copy()
        rm[m] = np.clip(row["gain"] * rd[m] + row["offset"], 0, 1)
        rend[tag], stats[tag] = rm, row["matched"]
    ys, xs = np.nonzero(fg)
    c, s = ((ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2), int(0.57 * max(np.ptp(ys), np.ptp(xs)))
    pad = s + 8
    P = lambda a: np.pad(a, pad)                                         # noqa: E731
    gt, fg = P(gt), P(fg)
    rend = {k: P(v) for k, v in rend.items()}
    c = (c[0] + pad, c[1] + pad)
    sy, sx = slice(c[0] - s, c[0] + s), slice(c[1] - s, c[1] + s)
    # zoom window: the square inside the body with the most image gradient energy
    gy, gx = np.gradient(cv2.GaussianBlur(gt, (0, 0), 1.0))
    e = (gy ** 2 + gx ** 2) * cv2.erode(fg.astype(np.uint8), np.ones((25, 25), np.uint8))
    zw = int(0.55 * s)                                                   # ~170 image pixels: stays legible
    ii = cv2.boxFilter(e, -1, (zw, zw), normalize=False)
    zy, zx = np.unravel_index(np.argmax(ii), ii.shape)
    zsy, zsx = slice(zy - zw // 2, zy + zw // 2), slice(zx - zw // 2, zx + zw // 2)

    W = rf.WMM * mm
    lab_w, cb_w, g, head = 4.6 * mm, 11.5 * mm, 0.45 * mm, 3.4 * mm
    pw = (W - lab_w - cb_w - 3 * g) / 4
    H = head + 3 * pw + 2 * g
    fig = plt.figure(figsize=(W, H), facecolor="white")

    def ax_at(x, y_top, w, h):
        ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
        ax.set_axis_off()
        return ax

    cols = [("gt", "Held-out image")] + METHODS
    imgs = {"gt": gt, **rend}
    for j, (k, t) in enumerate(cols):
        fig.text((lab_w + j * (pw + g) + pw / 2) / W, (H - head / 2) / H, t, ha="center", va="center", fontsize=6.5,
                 color=rf.INK, fontweight="bold" if k == "pcgs-v18n" else "normal")
    for i, t in enumerate(("Full view", "Zoom", "Error")):
        fig.text(lab_w * 0.42 / W, (H - head - i * (pw + g) - pw / 2) / H, t, rotation=90, ha="center",
                 va="center", fontsize=7, fontweight="bold", color=rf.INK)
    y = H - head
    # row 1: full views with the zoom box
    for j, (k, t) in enumerate(cols):
        ax = ax_at(lab_w + j * (pw + g), y, pw, pw)
        ax.imshow(imgs[k][sy, sx], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.add_patch(Rectangle((zsx.start - sx.start, zsy.start - sy.start), zw, zw, fill=False, ec=ZOOM, lw=0.7))
        if j == 0:
            rf.tile_text(ax, 0.03, 0.975, "a", size=8, bold=True)
            rf.tile_text(ax, 0.5, 0.018, "Hayabusa2 ONC-T, phase 19°", size=5.5, ha="center", va="bottom")
        else:
            st = stats[k]
            rf.tile_text(ax, 0.5, 0.018, f"PSNR {st['psnr']:.1f} dB   SSIM {st['ssim']:.3f}", size=5.5,
                         ha="center", va="bottom", bold=(k == "pcgs-v18n"))
    y -= pw + g
    # row 2: zooms
    for j, (k, t) in enumerate(cols):
        ax = ax_at(lab_w + j * (pw + g), y, pw, pw)
        ax.imshow(imgs[k][zsy, zsx], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=ZOOM, lw=1.2))
        if j == 0:
            rf.tile_text(ax, 0.03, 0.975, "b", size=8, bold=True, box=True)
    y -= pw + g
    # row 3: per-image PSNR (first tile) and residual maps
    fig.text((lab_w + 0.6 * mm) / W, (y - 3.0 * mm) / H, "c", fontsize=8, fontweight="bold", va="baseline",
             ha="left", color=rf.INK)
    axp = fig.add_axes([(lab_w + 9.0 * mm) / W, (y - pw + 4.2 * mm) / H, (pw - 10.0 * mm) / W, (pw - 15.6 * mm) / H])
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    axp.tick_params(length=2, width=0.5, pad=1.2, labelsize=5.5, colors=rf.INK)
    tags = [("pcgs-v18n", "PCGS"), ("pgsr", "PGSR"), ("ags", "AGS")]
    for bi, (body, bname) in enumerate(rf.BODIES):
        for ti, (tag, short) in enumerate(tags):
            v = [r["matched"]["psnr"] for s_ in (0, 1, 2) for r in per_image(d, body, tag, s_)]
            if not v:
                continue
            x = bi * 4 + ti + np.random.default_rng(ti).uniform(-0.2, 0.2, len(v))
            axp.scatter(x, v, s=2.2, color=rf.COLOR[tag], alpha=0.8, lw=0, zorder=3)
            axp.plot([bi * 4 + ti - 0.32, bi * 4 + ti + 0.32], [np.median(v)] * 2, color=rf.INK, lw=0.9, zorder=4)
    axp.set_xlim(-0.7, 6.7)
    axp.set_xticks([1, 5])
    axp.set_xticklabels([b for _, b in rf.BODIES], fontsize=6)
    axp.tick_params(axis="x", length=0)
    hl = [matplotlib.lines.Line2D([], [], ls="", marker="o", ms=2.6, mew=0, color=rf.COLOR[t], label=l)
          for t, l in (("pcgs-v18n", "PCGS (this work)"), ("pgsr", "PGSR"), ("ags", "Asteroid-GS"))]
    axp.legend(handles=hl, loc="lower left", bbox_to_anchor=(-0.32, 1.0), ncol=2, frameon=False, fontsize=5.5,
               handlelength=0.8, handletextpad=0.3, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.2)
    axp.set_ylabel("PSNR (dB)", fontsize=6, labelpad=1, color=rf.INK)
    cm = matplotlib.colormaps["PuOr_r"].copy()
    cm.set_bad("black")
    for j, (k, t) in enumerate(METHODS):
        ax = ax_at(lab_w + (j + 1) * (pw + g), y, pw, pw)
        r = np.where(fg, imgs[k] - gt, np.nan)[sy, sx]
        ax.imshow(r, cmap=cm, vmin=-RMAX, vmax=RMAX, interpolation="nearest")
        if j == 0:
            rf.tile_text(ax, 0.03, 0.975, "d", size=8, bold=True)
        rms = float(np.sqrt(np.nanmean(r ** 2)))
        rf.tile_text(ax, 0.5, 0.018, f"RMS residual {rms:.3f}", size=5.5, ha="center", va="bottom",
                     bold=(k == "pcgs-v18n"))
    cax = fig.add_axes([(W - cb_w + 1.4 * mm) / W, (y - 0.9 * pw) / H, 1.3 * mm / W, 0.8 * pw / H])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(-RMAX, RMAX), "PuOr_r"), cax=cax,
                      orientation="vertical")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
    cb.set_label("Render − image", fontsize=6, labelpad=1)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
    print("wrote", out, f"{H / mm:.0f} mm high")


if __name__ == "__main__":
    main()
