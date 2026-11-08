"""Figures of the A&A manuscript (paper/manuscript.tex): A&A page geometry (column 90 mm, page 184 mm), serif
type matching the text, white background, panel letters in parentheses. Inputs are small files fetched
from the server (aa_render.py panels, held-out renders, one image per campaign) and the metrics JSON.

  campaigns   one held-out image per campaign, ordered by phase angle
  shapes      held-out image, reference and three reconstructions in its view, with a zoom row
  errors      signed distance to the reference on the surface
  heldout     a held-out image, the renders of three methods and their residuals; PSNR against shape error
  phase       metrics against the phase angle for every campaign (PCGS variants and baselines)

usage: python aa_figs.py <kind> <input dir> <all_metrics.json> <out stem>
"""
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

mm = 1 / 25.4
COL, PAGE = 90 * mm, 184 * mm
INK, INK2 = "#1a1a1a", "#555555"
# one colour per method, the same in every figure
C = {"pcgs": "#c0392b", "pgsr": "#2e6fa7", "ags": "#3a9a5b", "2dgs": "#8c8c8c", "gof": "#b5a33c"}
LABEL = {"ref": "Reference", "pcgs": "PCGS", "pgsr": "PGSR", "ags": "Asteroid-GS", "2dgs": "2DGS", "gof": "GOF"}
CAMPAIGNS = [("ryugu_lp", "Ryugu", "ONC-T, 2019 Jan 15"), ("itokawa", "Itokawa", "AMICA, 2005 Sep 30"),
             ("ryugu", "Ryugu", "ONC-T, 2018 Jul 10"), ("ryugu_b2", "Ryugu", "ONC-T, 2018 Aug 31"),
             ("bennu", "Bennu", "MapCam, 2018 Dec 13")]
FOCAL = {"itokawa": 10066.7, "ryugu": 9269.2, "ryugu_lp": 9269.2, "ryugu_b2": 9269.2, "bennu": 14729.4}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "in", "ytick.direction": "in",
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "savefig.dpi": 600, "pdf.fonttype": 42, "legend.frameon": False})


def save(fig, out):
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".jpg", dpi=300, pil_kwargs=dict(quality=92))
    print("wrote", out)


def gray(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    return im.astype(np.float64) / (65535.0 if im.dtype == np.uint16 else 255.0)


def stretch(img, mask=None):
    v = img[mask] if mask is not None and mask.any() else img[img > 0.03]
    return np.clip(img / np.percentile(v, 99.7), 0, 1)


def square_crop(mask, margin=0.08):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = (1 + 2 * margin) * max(np.ptp(ys), np.ptp(xs)) / 2
    return int(round(cy - h)), int(round(cy + h)), int(round(cx - h)), int(round(cx + h))


def crop_pad(img, box):
    y0, y1, x0, x1 = box
    H, W = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, H), min(x1, W)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def scalebar(ax, n_px, m_per_px, length_m, color="white", y=0.06):
    """Scale bar of length_m metres in the lower left of an image axis of n_px pixels."""
    f = length_m / m_per_px / n_px
    ax.plot([0.06, 0.06 + f], [y, y], transform=ax.transAxes, color=color, lw=1.4, solid_capstyle="butt")
    ax.text(0.06 + f / 2, y + 0.03, f"{length_m:g} m", transform=ax.transAxes, color=color, ha="center",
            va="bottom", fontsize=6.5)


# ---------------------------------------------------------------- Fig. campaigns
def fig_campaigns(d, A, out):
    n = len(CAMPAIGNS)
    gap = 1.2 * mm
    w = (PAGE - (n - 1) * gap) / n
    fig = plt.figure(figsize=(PAGE, w + 9 * mm))
    for i, (key, body, inst) in enumerate(CAMPAIGNS):
        info = json.load(open(f"{d}/campaigns/{key}.json"))["info"]
        img, m = gray(f"{d}/campaigns/{key}.png"), gray(f"{d}/campaigns/{key}_mask.png") > 0.5
        box = square_crop(m)
        tile = crop_pad(stretch(img, m), box)
        ax = fig.add_axes([i * (w + gap) / PAGE, 7.5 * mm / (w + 9 * mm), w / PAGE, w / (w + 9 * mm)])
        ax.imshow(tile, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.set_axis_off()
        mpp = info["distance_km"] * 1000 / FOCAL[key]
        size = tile.shape[0] * mpp
        scalebar(ax, tile.shape[0], mpp, 100 if size < 900 else 200 if size < 1800 else 500)
        ax.text(0.04, 0.96, 'abcde'[i], transform=ax.transAxes, color="white", va="top", fontsize=9, fontweight="bold")
        ax.text(0.96, 0.96, f"$\\alpha={info['phase_deg']:.0f}^\\circ$", transform=ax.transAxes, color="white",
                ha="right", va="top", fontsize=8)
        fig.text((i * (w + gap) + w / 2) / PAGE, 4.2 * mm / (w + 9 * mm), body, ha="center", va="center", fontsize=8)
        fig.text((i * (w + gap) + w / 2) / PAGE, 1.3 * mm / (w + 9 * mm), inst, ha="center", va="center",
                 fontsize=7, color=INK2)
    save(fig, out)


# ---------------------------------------------------------------- Fig. held-out images
def per_image(d, body, tag, split=0):
    f = f"{d}/nvs/{body}__{tag}__s{split}.json"
    return json.load(open(f))["per_image"] if os.path.exists(f) else []


def fig_heldout(d, A, out, body="ryugu", idx=0):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paper_tables as pt
    tags = [("pcgs", "pcgs-v18n"), ("pgsr", "pgsr"), ("ags", "ags")]
    names = json.load(open(f"{d}/nvs/{body}_names.json"))
    stem = names[idx][:-4] if names[idx].endswith(".png") else names[idx]
    gt = gray(f"{d}/nvs/{body}_{idx:02d}_gt.png")
    fg = gray(f"{d}/nvs/{body}_{idx:02d}_mask.png") > 0.5
    box = square_crop(fg, 0.05)
    hi = np.percentile(gt[fg], 99.7)
    tiles, res, stats = [crop_pad(np.clip(gt / hi, 0, 1), box)], [], []
    for k, t in tags:
        rd = gray(f"{d}/nvs/{body}_{idx:02d}_{t}.png")
        row = next(r for r in per_image(d, body, t) if r["image"] in (f"{idx:05d}", stem))
        m = fg | (rd > 2.0 / 255)
        rm = rd.copy()
        rm[m] = np.clip(row["gain"] * rd[m] + row["offset"], 0, 1)
        tiles.append(crop_pad(np.clip(rm / hi, 0, 1), box))
        r = np.where(fg, rm - gt, np.nan)
        res.append(crop_pad(r, box))
        stats.append(row["matched"])
    # layout: 4 image columns, residual row under the three renders, scatter on the right
    tw, gap = 31 * mm, 1.0 * mm
    sw = PAGE - 4 * tw - 3 * gap - 15 * mm
    H = 2 * tw + gap + 12 * mm
    fig = plt.figure(figsize=(PAGE, H))
    fy = lambda y: y / H  # noqa: E731
    titles = ["Observed"] + [LABEL[k] for k, _ in tags]
    for j in range(4):
        ax = fig.add_axes([j * (tw + gap) / PAGE, fy(tw + gap + 6 * mm), tw / PAGE, fy(tw)])
        ax.imshow(tiles[j], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(titles[j], fontsize=8, pad=2)
        if j:
            ax.text(0.04, 0.04, f"{stats[j - 1]['psnr']:.1f} dB", transform=ax.transAxes, color="white", fontsize=7)
        if j == 0:
            ax.text(0.04, 0.96, "(a)", transform=ax.transAxes, color="white", va="top")
    norm = Normalize(-0.15, 0.15)
    for j in range(3):
        ax = fig.add_axes([(j + 1) * (tw + gap) / PAGE, fy(6 * mm), tw / PAGE, fy(tw)])
        ax.imshow(np.zeros(res[j].shape), cmap="gray", vmin=0, vmax=1)
        ax.imshow(res[j], cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_axis_off()
        rms = np.sqrt(np.nanmean(res[j] ** 2))
        ax.text(0.04, 0.04, f"rms {rms:.3f}", transform=ax.transAxes, color="white", fontsize=7)
        if j == 0:
            ax.text(0.04, 0.96, "(b)", transform=ax.transAxes, color="white", va="top")
    cax = fig.add_axes([0.6 * tw / PAGE, fy(6 * mm + 0.15 * tw), 0.05 * tw / PAGE, fy(0.7 * tw)])
    cb = fig.colorbar(ScalarMappable(norm, "RdBu_r"), cax=cax)
    cb.set_label("Render $-$ image", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    # (c) held-out PSNR against mean distance, three-split means
    ax = fig.add_axes([(4 * tw + 3 * gap + 13 * mm) / PAGE, fy(9 * mm), sw / PAGE, fy(H - 15 * mm)])
    rows = [("pcgs", "pcgs-v18n"), ("pgsr", "pgsr"), ("ags", "ags"), ("2dgs", "2dgs-dtu"), ("gof", "gof")]
    for b, mk in (("itokawa", "o"), ("ryugu", "s")):
        for k, t in rows:
            x = pt.values(A, pt.row("", b, t), pt.col("psnr"))
            y = pt.values(A, pt.row("", b, t), pt.col("mean"))
            if x and y:
                ax.scatter(np.mean(x), np.mean(y), s=22, marker=mk, color=C[k], edgecolor="white", lw=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Held-out PSNR (dB)")
    ax.set_ylabel("Mean distance (m)")
    ax.set_yticks([0.5, 1, 2, 5])
    ax.set_yticklabels(["0.5", "1", "2", "5"])
    ax.set_ylim(0.45, 9)
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes, va="top")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=4, color=C[k], label=LABEL[k]) for k, _ in rows]
    h += [Line2D([], [], ls="", marker="o", ms=4, mfc="none", color=INK2, label="Itokawa"),
          Line2D([], [], ls="", marker="s", ms=4, mfc="none", color=INK2, label="Ryugu")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=6.5, handletextpad=0.2,
              labelspacing=0.3, borderaxespad=0.3)
    save(fig, out)


# ---------------------------------------------------------------- Fig. overview (method)
def fig_overview(d, A, out, view="hyb2_onc_20180710_060508_tvf_l2c"):
    from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
    z = np.load(f"{d}/decomp/{view}.npz")
    alpha = z["alpha"].astype(np.float32)
    box = square_crop(alpha > 0.5, 0.04)
    hi = np.percentile(z["gt"].astype(np.float32)[alpha > 0.5], 99.7)
    th = {"image": crop_pad(np.clip(z["gt"].astype(np.float32) / hi, 0, 1), box),
          "albedo": crop_pad(np.clip(z["albedo"].astype(np.float32) / np.percentile(z["albedo"][alpha > 0.5], 99.5), 0, 1), box),
          "phi": crop_pad(np.clip(z["phi"].astype(np.float32), 0, 1), box),
          "normal": crop_pad(np.clip((np.moveaxis(z["normal"].astype(np.float32), 0, -1) * [1, -1, -1] + 1) / 2, 0, 1)
                             * (alpha[..., None] > 0.5), box),
          "pred": crop_pad(np.clip(z["pred"].astype(np.float32) / hi, 0, 1), box)}
    W, H = 184, 78
    fig = plt.figure(figsize=(W * mm, H * mm))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.set_axis_off()

    def panel(x, y, w, h, title):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc="#f4f3f1",
                                    ec="#9a9a9a", lw=0.6))
        bg.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=8, weight="bold")

    def thumb(img, x, y, s, label=None, cmap="gray"):
        ax = fig.add_axes([x / W, y / H, s / W, s / H])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="lanczos")
        ax.set_axis_off()
        if label:
            bg.text(x + s / 2, y - 1.6, label, ha="center", va="top", fontsize=7)

    def arrow(p, q, rad=0.0, label=None, lpos=None):
        bg.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
        if label:
            bg.text(*lpos, label, ha="center", va="center", fontsize=6.8, color=INK2, style="italic")

    top, th_ = 32, 38
    on = (alpha > 0.5)
    for k in ("albedo", "phi"):
        th[k] = th[k] * crop_pad(on.astype(np.float32), box)
    # (1) inputs
    panel(2, top, 32, th_, "Input")
    thumb(th["image"], 8, top + 12, 20)
    bg.text(18, top + 7.6, "50 images, SfM poses", ha="center", fontsize=7)
    bg.text(18, top + 4.2, "Sun direction $\\mathbf{s}$ (SPICE)", ha="center", fontsize=7)
    # (2) surfels
    panel(40, top, 30, th_, "2D Gaussian surfels")
    rng = np.random.default_rng(3)
    for k in range(11):
        cx, cy = 45 + 20 * rng.random(), top + 14 + 13 * rng.random()
        bg.add_patch(Ellipse((cx, cy), 7 + 3 * rng.random(), 2 + 1.5 * rng.random(), angle=180 * rng.random(),
                             fc=plt.cm.Greys(0.35 + 0.4 * rng.random()), ec="#555555", lw=0.4, alpha=0.85))
    bg.text(55, top + 7.6, "centre, axes, opacity", ha="center", fontsize=7)
    bg.text(55, top + 4.2, "albedo $a_i$ (view-independent)", ha="center", fontsize=7)
    # (3) rendering under the known illumination
    panel(76, top, 68, th_, "Rendering under the known illumination")
    s, y0 = 12, top + 15
    thumb(th["albedo"], 78.0, y0, s, "albedo $A$")
    bg.text(91.6, y0 + s / 2, "$\\times$", ha="center", va="center", fontsize=9)
    thumb(th["phi"], 93.2, y0, s, "shading $\\Phi$")
    bg.text(111.1, y0 + s / 2, "$\\times\\,V\\,=$", ha="center", va="center", fontsize=8.5)
    thumb(th["pred"], 117.0, y0, s, "render $\\hat I$")
    thumb(th["normal"], 131.6, y0, s, "normal $\\mathbf{n}$")
    bg.plot([130.3, 130.3], [y0 - 4, y0 + s + 1], color="#b0b0b0", lw=0.5)
    bg.text(110, top + 7.0, "$\\Phi$: lunar-Lambert function of $\\mu_0=\\mathbf{n}\\cdot\\mathbf{s}$ and "
            "$\\mu=\\mathbf{n}\\cdot\\mathbf{v}$", ha="center", fontsize=7)
    bg.text(110, top + 3.6, "$V$: cast shadows from a Sun-view depth map", ha="center", fontsize=7)
    # (4) objective
    panel(150, top, 32, th_, "Training objective")
    lines = ["image: $\\mathcal{L}_1$ + D-SSIM", "silhouette, night side", "   behind the limb excluded",
             "multi-view NCC", "phase-dependent prior:", "   $\\alpha<15^\\circ$: albedo TV",
             "   $\\alpha\\geq15^\\circ$: cast shadows"]
    for i, t in enumerate(lines):
        bg.text(152.5, top + th_ - 8.5 - 4.0 * i, t, ha="left", va="center", fontsize=7)
    arrow((34, top + th_ / 2), (40, top + th_ / 2))
    arrow((70, top + th_ / 2), (76, top + th_ / 2))
    arrow((144, top + th_ / 2), (150, top + th_ / 2))
    # gradient feedback above the panels
    yf = top + th_ + 3.2
    bg.plot([166, 166, 55], [top + th_, yf, yf], color=INK2, lw=0.8, solid_joinstyle="miter")
    bg.add_patch(FancyArrowPatch((55, yf), (55, top + th_), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color=INK2, shrinkA=0, shrinkB=0))
    bg.text(110, yf + 2.2, "gradients update the geometry and the albedo (30 000 iterations)", ha="center",
            fontsize=6.8, color=INK2, style="italic")
    # (5) extraction
    panel(2, 3, 180, 24, "Reference-free surface extraction")
    steps = ["median depth and\nnormal per view", "keep depth confirmed\nby $\\geq$3 other views",
             "screened Poisson\nreconstruction", "fair regions that\nno view supports", "offset to the\nSfM tie points"]
    bx, bw, gap = 8, 25, 6
    for i, t in enumerate(steps):
        x = bx + i * (bw + gap)
        bg.add_patch(FancyBboxPatch((x, 6.5), bw, 12, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                    ec="#9a9a9a", lw=0.5))
        bg.text(x + bw / 2, 12.5, t, ha="center", va="center", fontsize=6.8, linespacing=1.15)
        if i:
            arrow((x - gap, 12.5), (x, 12.5))
    xe = bx + 5 * (bw + gap) - gap
    if os.path.exists(f"{d}/ryugu_pcgs_clay.png"):
        c = cv2.imread(f"{d}/ryugu_pcgs_clay.png", cv2.IMREAD_UNCHANGED)
        c = cv2.cvtColor(c, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
        rgb = c[..., :3] * c[..., 3:] + (1 - c[..., 3:]) * np.array([0.957, 0.953, 0.945])
        cb = square_crop(c[..., 3] > 0.5, 0.03)
        ax = fig.add_axes([(xe + 6) / W, 4.0 / H, 18 / W, 18 / H])
        ax.imshow(crop_pad(rgb, cb), interpolation="lanczos")
        ax.set_axis_off()
        bg.text(xe + 15, 3.6, "watertight mesh", ha="center", va="top", fontsize=6.5)
    arrow((xe, 12.5), (xe + 6, 12.5))
    arrow((55, top), (55, 27), label="trained surfels", lpos=(64, 29.6))
    save(fig, out)


# ---------------------------------------------------------------- Figs. shapes and errors (render panels)
BODIES3 = [("itokawa", "Itokawa"), ("ryugu", "Ryugu"), ("bennu", "Bennu")]
KEYS = ["ref", "pcgs", "pgsr", "ags"]
ZOOM_ASPECT = 1.6
ZC = "#e08a1e"
PAD = 400


def panel_img(d, name, sub="panels"):
    p = f"{d}/{sub}/{name}"
    if os.path.exists(p + ".png"):
        a = cv2.imread(p + ".png", cv2.IMREAD_UNCHANGED)
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA if a.ndim == 3 and a.shape[2] == 4 else cv2.COLOR_BGR2RGB) \
            if a.ndim == 3 else a
        return a.astype(np.float32) / 255
    a = np.load(p + ".npy")
    return a.astype(np.float32) / (255 if a.dtype == np.uint8 else 1)


def on_white(rgba):
    return np.clip(rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]), 0, 1)


def body_block(d, body, mt):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nc_fig_recon import img_affine
    ii = mt["image"]
    pad = lambda a: np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))  # noqa: E731
    clay = {k: pad(panel_img(d, f"{body}_{k}_clay")) for k in KEYS}
    err = {k: pad(panel_img(d, f"{body}_{k}_err")) for k in KEYS[1:]}
    a = np.zeros(clay["ref"].shape[:2], bool)
    for v in list(clay.values()) + list(err.values()):
        a |= v[..., 3] > 0.02
    ys, xs = np.nonzero(a)
    m = int(0.03 * max(np.ptp(ys), np.ptp(xs)))
    sy, sx = slice(ys.min() - m, ys.max() + m), slice(xs.min() - m, xs.max() + m)
    sr = 2 * mt["half"] / mt["res"]
    x0, y0 = -mt["half"] + (sx.start - PAD) * sr, mt["half"] - (sy.start - PAD) * sr
    im = panel_img(d, f"{body}_image")
    im = im[..., 0] if im.ndim == 3 else im
    full = cv2.warpAffine(im, img_affine(ii, mt["rot"], x0, y0, sr), (sx.stop - sx.start, sy.stop - sy.start),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    zx, zy, zw = mt["zoom"]
    zoom = {k: panel_img(d, f"{body}_{k}_zoom") for k in KEYS}
    n = zoom["ref"].shape[0]
    nb = int(round(n / ZOOM_ASPECT))
    yo = (n - nb) // 2
    zoom = {k: on_white(v[yo:yo + nb]) for k, v in zoom.items()}
    step = zw / n
    zimg = cv2.warpAffine(im, img_affine(ii, mt["rot"], zx - zw / 2, zy + zw / 2 - yo * step, step), (n, nb),
                          flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    hi = np.percentile(full[full > 0.03], 99.6)
    box = ((zx - zw / 2 - x0) / sr, (y0 - (zy + zw / ZOOM_ASPECT / 2)) / sr, zw / sr, zw / ZOOM_ASPECT / sr)
    hist = {k: np.load(f"{d}/panels/{body}_{k}_sd_hist.npy") for k in KEYS[1:]
            if os.path.exists(f"{d}/panels/{body}_{k}_sd_hist.npy")}
    return dict(image=np.clip(full / hi, 0, 1), zimage=np.clip(zimg / hi, 0, 1),
                clay={k: on_white(v[sy, sx]) for k, v in clay.items()},
                err={k: on_white(v[sy, sx]) for k, v in err.items()}, zoom=zoom, box=box,
                asp=(sx.stop - sx.start) / (sy.stop - sy.start), sr=sr, zstep=step, hist=hist, stats=mt["stats"])


def fig_shapes(d, A, out):
    from matplotlib.patches import Rectangle
    meta = json.load(open(f"{d}/panels/meta.json"))
    blocks = [(b, name, body_block(d, b, meta[b])) for b, name in BODIES3 if b in meta and "image" in meta[b]]
    lab, g, gb, head = 5.0 * mm, 0.7 * mm, 2.4 * mm, 4.2 * mm
    pw = (PAGE - lab - 4 * g) / 5
    hz = pw / ZOOM_ASPECT
    H = head + sum(pw / bl["asp"] + hz + g for _, _, bl in blocks) + gb * (len(blocks) - 1)
    fig = plt.figure(figsize=(PAGE, H))
    titles = ["Image", "Reference", "PCGS", "PGSR", "Asteroid-GS"]
    y = H - head
    bars = {"itokawa": (100, 20), "ryugu": (200, 50), "bennu": (100, 20)}
    for bi, (b, name, bl) in enumerate(blocks):
        hf = pw / bl["asp"]
        for j in range(5):
            x = lab + j * (pw + g)
            ax = fig.add_axes([x / PAGE, (y - hf) / H, pw / PAGE, hf / H])
            ax.imshow(bl["image"] if j == 0 else bl["clay"][KEYS[j - 1]], cmap="gray", vmin=0, vmax=1,
                      interpolation="lanczos", aspect="auto")
            ax.set_axis_off()
            if j in (0, 1):
                ax.add_patch(Rectangle(bl["box"][:2], *bl["box"][2:], fill=False, ec=ZC, lw=0.7))
            if j == 1:
                L = bars[b][0]
                nx = bl["clay"]["ref"].shape[1]
                f = L / bl["sr"] / nx
                ax.plot([0.05, 0.05 + f], [0.07, 0.07], transform=ax.transAxes, color=INK, lw=1.2)
                ax.text(0.05 + f / 2, 0.1, f"{L} m", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)
            if bi == 0:
                ax.set_title(titles[j], fontsize=8, pad=2.5)
            ax2 = fig.add_axes([x / PAGE, (y - hf - g - hz) / H, pw / PAGE, hz / H])
            ax2.imshow(bl["zimage"] if j == 0 else bl["zoom"][KEYS[j - 1]], cmap="gray", vmin=0, vmax=1,
                       interpolation="lanczos", aspect="auto")
            ax2.set_axis_off()
            ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes, fill=False, ec=ZC, lw=0.9))
            if j == 1:
                L = bars[b][1]
                f = L / bl["zstep"] / bl["zoom"]["ref"].shape[1]
                ax2.plot([0.05, 0.05 + f], [0.09, 0.09], transform=ax2.transAxes, color=INK, lw=1.2)
                ax2.text(0.05 + f / 2, 0.13, f"{L} m", transform=ax2.transAxes, ha="center", va="bottom",
                         fontsize=6.5)
        fig.text(0.45 * lab / PAGE, (y - (hf + g + hz) / 2) / H, name, rotation=90, ha="center", va="center",
                 fontsize=8)
        y -= hf + g + hz + gb
    save(fig, out)


def fig_errors(d, A, out):
    from matplotlib.cm import ScalarMappable
    meta = json.load(open(f"{d}/panels/meta.json"))
    blocks = [(b, name, body_block(d, b, meta[b])) for b, name in BODIES3 if b in meta and "image" in meta[b]]
    lab, g, gb, head, hw = 5.0 * mm, 0.7 * mm, 1.6 * mm, 4.2 * mm, 40 * mm
    pw = (PAGE - lab - 3 * g - hw - 9 * mm) / 3
    H = head + sum(pw / bl["asp"] + gb for _, _, bl in blocks) + 14 * mm
    fig = plt.figure(figsize=(PAGE, H))
    y = H - head
    vmax = 3.0
    for bi, (b, name, bl) in enumerate(blocks):
        hf = pw / bl["asp"]
        for j, k in enumerate(KEYS[1:]):
            x = lab + j * (pw + g)
            ax = fig.add_axes([x / PAGE, (y - hf) / H, pw / PAGE, hf / H])
            ax.imshow(bl["err"][k], interpolation="lanczos", aspect="auto")
            ax.set_axis_off()
            st = bl["stats"].get(k, {})
            if st:
                ax.text(0.03, 0.04, f"mean $|d|$ {st['mean_abs']:.2f} m", transform=ax.transAxes, fontsize=6.5)
            if bi == 0:
                ax.set_title(LABEL[k], fontsize=8, pad=2.5)
        hx = lab + 3 * (pw + g) + 9 * mm
        ax = fig.add_axes([hx / PAGE, (y - hf + 0.12 * hf) / H, hw / PAGE, 0.8 * hf / H])
        edges = np.linspace(-6, 6, 241)
        c = 0.5 * (edges[1:] + edges[:-1])
        for k in KEYS[1:]:
            h = bl["hist"].get(k)
            if h is not None:
                ax.plot(c, h / h.sum() / (edges[1] - edges[0]), color=C[k], lw=0.9, label=LABEL[k])
        ax.set_xlim(-vmax, vmax)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6.5)
        if bi == len(blocks) - 1:
            ax.set_xlabel("Signed distance $d$ (m)", fontsize=7)
        if bi == 0:
            ax.legend(fontsize=6.3, loc="upper right", handlelength=1.2, borderaxespad=0.2)
        ax.set_ylabel("Density", fontsize=7)
        fig.text(0.45 * lab / PAGE, (y - hf / 2) / H, name, rotation=90, ha="center", va="center", fontsize=8)
        y -= hf + gb
    cax = fig.add_axes([(lab + 0.6 * pw) / PAGE, 9.0 * mm / H, (1.8 * pw) / PAGE, 1.8 * mm / H])
    cb = fig.colorbar(ScalarMappable(Normalize(-vmax, vmax), "RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_label("Signed distance to the reference (m)", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, out)


def nc_tile(ax, img, cmap=None):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="lanczos", aspect="auto")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def on_black_rgba(rgba):
    return np.clip(rgba[..., :3] * rgba[..., 3:], 0, 1)


def crop_like(block_d, body, mt, extra):
    """The crop window of body_block (union of the reference and method silhouettes) applied to extra panels."""
    pad = lambda a: np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))  # noqa: E731
    clay = [pad(panel_img(block_d, f"{body}_{k}_clay")) for k in KEYS] + [pad(e) for e in extra]
    a = np.zeros(clay[0].shape[:2], bool)
    for v in clay:
        a |= v[..., 3] > 0.02
    ys, xs = np.nonzero(a)
    m = int(0.03 * max(np.ptp(ys), np.ptp(xs)))
    return slice(ys.min() - m, ys.max() + m), slice(xs.min() - m, xs.max() + m), pad


def letter(fig, x, y, W, H, s, color="black"):
    fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", va="top", ha="left", color=color)


def fig_night(d, A, out):
    """Night-side band: what the image mask misses on Bennu, and PCGS without and with the band."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_figs as rf
    from nc_fig_recon import img_affine
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rf.style()
    body = "bennu"
    mt = json.load(open(f"{d}/panels/meta.json"))[body]
    p2 = f"{d}/panels2"
    noband = {k: panel_img(d, f"bennu_night_pcgsnoband_{k}", "panels2") for k in ("clay", "err")}
    sy, sx, pad = crop_like(d, body, mt, [noband["clay"]])
    sr = 2 * mt["half"] / mt["res"]
    x0, y0 = -mt["half"] + (sx.start - PAD) * sr, mt["half"] - (sy.start - PAD) * sr
    shape = (sx.stop - sx.start, sy.stop - sy.start)
    band = np.load(f"{d}/band/bennu.npz")
    Af = img_affine(mt["image"], mt["rot"], x0, y0, sr)
    warp = lambda a, fl: cv2.warpAffine(a.astype(np.float32), Af, shape, flags=fl | cv2.WARP_INVERSE_MAP)  # noqa: E731
    img = warp(band["image"] / 255.0, cv2.INTER_LINEAR)
    img = np.clip(img / np.percentile(img[img > 0.03], 99.6), 0, 1)
    lit = warp(band["mask"], cv2.INTER_NEAREST) > 0.5
    nb = warp(band["band"], cv2.INTER_NEAREST) > 0.5
    ref_a = pad(panel_img(d, f"{body}_ref_clay"))[sy, sx][..., 3] > 0.5
    tiles = [on_black_rgba(pad(panel_img(d, f"{body}_ref_clay"))[sy, sx]), on_black_rgba(pad(noband["clay"])[sy, sx]),
             on_black_rgba(pad(panel_img(d, f"{body}_pcgs_clay"))[sy, sx])]
    errs = [on_black_rgba(pad(noband["err"])[sy, sx]), on_black_rgba(pad(panel_img(d, f"{body}_pcgs_err"))[sy, sx])]
    st2 = json.load(open(f"{p2}/meta.json"))["bennu_night"]["stats"]["pcgsnoband"]
    st = mt["stats"]["pcgs"]
    W, g, cbw = 183 * mm, 0.6 * mm, 7.0 * mm
    pw = (W - 3 * g - cbw) / 4
    ph = pw * shape[1] / shape[0]
    head = 4.6 * mm
    H = head + 2 * ph + g + 1.0 * mm
    fig = plt.figure(figsize=(W, H), facecolor="white")
    titles = [f"Bennu ({mt['image']['phase']:.0f}°): foreground mask and night-side band", "Reference (OLA)", "PCGS without night-side band", "PCGS (this work)"]
    for j in range(4):
        ax = fig.add_axes([j * (pw + g) / W, (H - head - ph) / H, pw / W, ph / H])
        if j == 0:
            rgb = np.repeat(img[..., None], 3, -1)
            ov = rgb.copy()
            ov[nb] = 0.55 * ov[nb] + 0.45 * np.array([0.25, 0.55, 1.0])
            nc_tile(ax, ov)
            ax.contour(lit.astype(float), [0.5], colors=["#F0E442"], linewidths=0.7)
            ax.contour(ref_a.astype(float), [0.5], colors=["white"], linewidths=0.7, linestyles="--")
            ax.legend(handles=[Line2D([], [], color="#F0E442", lw=1, label="foreground mask"),
                               Patch(fc=(0.25, 0.55, 1.0), alpha=0.6, label="night-side band"),
                               Line2D([], [], color="white", lw=1, ls="--", label="body (reference)")],
                      loc="lower left", fontsize=5.3, frameon=True, facecolor="black", framealpha=0.65, edgecolor="none",
                      labelcolor="white", handlelength=1.4, borderpad=0.4)
        else:
            nc_tile(ax, tiles[j - 1])
        ax.set_title(titles[j], fontsize=6.5, pad=2, fontweight="bold" if j == 3 else "normal")
    letter(fig, 0.8 * mm, H - head - 0.6 * mm, W, H, "a", "white")
    # row 2: distributions, the Ryugu 40 deg example, error maps
    y2 = H - head - ph - g - ph
    axd = fig.add_axes([(0 + 9 * mm) / W, (y2 + 7.5 * mm) / H, (pw - 11 * mm) / W, (ph - 11 * mm) / H])
    edges = np.linspace(-6, 6, 241)
    c = 0.5 * (edges[1:] + edges[:-1])
    for f, col, lw, lab_ in ((f"{p2}/bennu_night_pcgsnoband_sd_hist.npy", rf.COLOR["pcgs-v18"], 1.0,
                              "without band"),
                             (f"{d}/panels/bennu_pcgs_sd_hist.npy", rf.COLOR["pcgs-v18n"], 1.3, "PCGS")):
        h = np.load(f)
        far = 100 * (h[0] + h[-1]) / h.sum()           # the end bins collect |d| beyond 6 m (clipped)
        axd.plot(c[1:-1], h[1:-1] / h.sum() / (edges[1] - edges[0]), color=col, lw=lw,
                 label=f"{lab_}: {far:.0f}% at |d| > 6 m")
    axd.axvline(0, ymax=0.78, color=INK2, lw=0.5, ls=(0, (2, 1.5)))   # stops below the legend
    axd.set_xlim(-6, 6)
    axd.set_ylim(0, 1.45 * axd.get_ylim()[1])
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    axd.tick_params(labelsize=5.5, length=2, width=0.5)
    axd.set_xlabel("Signed distance d (m)", fontsize=6, labelpad=1)
    axd.set_ylabel("Density (1/m)", fontsize=6, labelpad=1)
    axd.legend(fontsize=5.0, frameon=False, loc="upper left", handlelength=1.1, bbox_to_anchor=(-0.02, 1.08))
    letter(fig, 0.6 * mm, y2 + ph - 0.4 * mm, W, H, "b")
    rb = np.load(f"{d}/band/ryugu_b2.npz")
    m = rb["mask"]
    bx = square_crop(m | rb["band"], 0.06)
    im2 = crop_pad(np.clip(rb["image"] / np.percentile(rb["image"][m], 99.6), 0, 1), bx)
    nb2, lit2 = crop_pad(rb["band"], bx), crop_pad(m, bx)
    ov2 = np.repeat(im2[..., None], 3, -1)
    ov2[nb2] = 0.55 * ov2[nb2] + 0.45 * np.array([0.25, 0.55, 1.0])
    ax = fig.add_axes([(pw + g) / W, y2 / H, pw / W, ph / H])
    nc_tile(ax, ov2)
    ax.contour(lit2.astype(float), [0.5], colors=["#F0E442"], linewidths=0.7)
    ax.text(0.97, 0.03, f"Ryugu, ONC-T, phase {float(rb['phase']):.0f}°", transform=ax.transAxes, color="white",
            fontsize=5.8, va="bottom", ha="right")
    letter(fig, (pw + g) + 0.8 * mm, y2 + ph - 0.6 * mm, W, H, "c", "white")
    for j, (e, s_) in enumerate(((errs[0], st2), (errs[1], st))):
        ax = fig.add_axes([(2 + j) * (pw + g) / W, y2 / H, pw / W, ph / H])
        nc_tile(ax, e)
        ax.text(0.5, 0.02, f"mean |d| {s_['mean_abs']:.2f} m   RMSE {s_['rmse']:.2f} m", transform=ax.transAxes,
                color="white", fontsize=5.8, ha="center", va="bottom", fontweight="bold" if j else "normal")
        if j == 0:
            letter(fig, 2 * (pw + g) + 0.8 * mm, y2 + ph - 0.6 * mm, W, H, "d", "white")
    cax = fig.add_axes([(4 * pw + 3 * g + 1.2 * mm) / W, (y2 + 0.15 * ph) / H, 1.0 * mm / W, 0.7 * ph / H])
    cb = fig.colorbar(ScalarMappable(Normalize(-3, 3), "RdBu_r"), cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=0.5)
    cb.set_label("d (m)", fontsize=5.5, labelpad=1)
    cb.outline.set_linewidth(0.4)
    fig.savefig(out + ".png", dpi=600, facecolor="white")
    fig.savefig(out + ".pdf", facecolor="white")
    print("wrote", out)


def fig_ryugu_phase(d, A, out):
    """The same body at three phase angles: Ryugu reconstructed from the 7, 19 and 40 deg campaigns (and
    Asteroid-GS at 40 deg), seen from the camera of the 19 deg held-out image of Fig. 2."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_figs as rf
    from matplotlib.patches import Rectangle
    rf.style()
    body = "ryugu"
    mt = json.load(open(f"{d}/panels/meta.json"))[body]
    p2 = f"{d}/panels2"
    st = json.load(open(f"{p2}/meta.json"))["ryugu_phase"]["stats"]
    keys = [("pcgs07", "PCGS, 7°"), ("pcgs19", "PCGS, 19°"), ("pcgs40", "PCGS, 40°"), ("ags40", "Asteroid-GS, 40°")]
    extra = {k: {t: panel_img(d, f"ryugu_phase_{k}_{t}", "panels2") for t in ("clay", "zoom", "err")} for k, _ in keys}
    sy, sx, pad = crop_like(d, body, mt, [extra[k]["clay"] for k, _ in keys])
    shape = (sx.stop - sx.start, sy.stop - sy.start)
    sr = 2 * mt["half"] / mt["res"]
    x0, y0 = -mt["half"] + (sx.start - PAD) * sr, mt["half"] - (sy.start - PAD) * sr
    zx, zy, zw = mt["zoom"]
    box = ((zx - zw / 2 - x0) / sr, (y0 - (zy + zw / ZOOM_ASPECT / 2)) / sr, zw / sr, zw / ZOOM_ASPECT / sr)
    n = panel_img(d, f"{body}_ref_zoom").shape[0]
    nb = int(round(n / ZOOM_ASPECT))
    yo = (n - nb) // 2
    cols = [("ref", "Reference (SfM-MVS)")] + keys
    full = {"ref": on_black_rgba(pad(panel_img(d, f"{body}_ref_clay"))[sy, sx])}
    zoom = {"ref": on_black_rgba(panel_img(d, f"{body}_ref_zoom")[yo:yo + nb])}
    err = {}
    for k, _ in keys:
        full[k] = on_black_rgba(pad(extra[k]["clay"])[sy, sx])
        zoom[k] = on_black_rgba(extra[k]["zoom"][yo:yo + nb])
        err[k] = on_black_rgba(pad(extra[k]["err"])[sy, sx])
    W, g = 183 * mm, 0.6 * mm
    pw = (W - 4 * g - 7.0 * mm) / 5
    ph, hz, head = pw * shape[1] / shape[0], pw / ZOOM_ASPECT, 4.6 * mm
    H = head + 2 * ph + hz + 2 * g + 0.6 * mm
    fig = plt.figure(figsize=(W, H), facecolor="white")
    y = H - head
    for j, (k, t) in enumerate(cols):
        x = j * (pw + g)
        ax = fig.add_axes([x / W, (y - ph) / H, pw / W, ph / H])
        nc_tile(ax, full[k])
        if k == "ref":
            ax.add_patch(Rectangle(box[:2], *box[2:], fill=False, ec=rf.ZOOM, lw=0.6))
        ax.set_title(t, fontsize=6.5, pad=2, fontweight="bold" if k.startswith("pcgs") else "normal")
        ax2 = fig.add_axes([x / W, (y - ph - g - hz) / H, pw / W, hz / H])
        nc_tile(ax2, zoom[k])
        ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes, fill=False, ec=rf.ZOOM, lw=0.8))
        ax3 = fig.add_axes([x / W, (y - 2 * ph - 2 * g - hz) / H, pw / W, ph / H])
        if k == "ref":
            axd = fig.add_axes([(x + 7.5 * mm) / W, (y - 2 * ph - 2 * g - hz + 7 * mm) / H, (pw - 8.5 * mm) / W,
                                (ph - 10 * mm) / H])
            ax3.set_axis_off()
            edges = np.linspace(-6, 6, 241)
            c = 0.5 * (edges[1:] + edges[:-1])
            for kk, col in (("pcgs07", "#F2A93B"), ("pcgs19", rf.COLOR["pcgs-v18n"]), ("pcgs40", "#7A2E0B"),
                            ("ags40", rf.COLOR["ags"])):
                h = np.load(f"{p2}/ryugu_phase_{kk}_sd_hist.npy")
                axd.plot(c, h / h.sum() / (edges[1] - edges[0]), color=col, lw=1.0, label=dict(keys)[kk])
            axd.set_xlim(-3, 3)
            axd.set_ylim(0, 1.75 * axd.get_ylim()[1])
            for sp in ("top", "right"):
                axd.spines[sp].set_visible(False)
            axd.tick_params(labelsize=5.2, length=2, width=0.5)
            axd.set_xlabel("Signed distance d (m)", fontsize=5.8, labelpad=1)
            axd.set_ylabel("Density (1/m)", fontsize=5.8, labelpad=1)
            axd.legend(fontsize=4.8, frameon=False, loc="upper left", handlelength=1.1, labelspacing=0.15, ncol=1,
                       columnspacing=0.6)
        else:
            nc_tile(ax3, err[k])
            s_ = st[k]
            ax3.text(0.5, 0.02, f"mean |d| {s_['mean_abs']:.2f} m   RMSE {s_['rmse']:.2f} m", transform=ax3.transAxes,
                     color="white", fontsize=5.6, ha="center", va="bottom",
                     fontweight="bold" if k.startswith("pcgs") else "normal")
    for lt, yy, col in (("a", y, "white"), ("b", y - ph - g, "white"), ("c", y - ph - 2 * g - hz, "black")):
        letter(fig, 0.8 * mm, yy - 0.6 * mm, W, H, lt, col)
    cax = fig.add_axes([(5 * pw + 4 * g + 1.2 * mm) / W, (y - 2 * ph - 2 * g - hz + 0.15 * ph) / H, 1.0 * mm / W,
                        0.7 * ph / H])
    cb = fig.colorbar(ScalarMappable(Normalize(-3, 3), "RdBu_r"), cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=0.5)
    cb.set_label("d (m)", fontsize=5.5, labelpad=1)
    cb.outline.set_linewidth(0.4)
    fig.savefig(out + ".png", dpi=600, facecolor="white")
    fig.savefig(out + ".pdf", facecolor="white")
    print("wrote", out)


def fig_teaser(d, A, out):
    """Project-page teaser in the style of the NC figures: per body one row of full views on black
    (held-out image, reference, PCGS, PGSR, Asteroid-GS), sans-serif labels."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_figs as rf
    rf.style()
    meta = json.load(open(f"{d}/panels/meta.json"))
    blocks = [(b, name, body_block(d, b, meta[b])) for b, name in BODIES3 if b in meta and "image" in meta[b]]
    lab, g, head, W = 6.0 * mm, 0.5 * mm, 5.0 * mm, 183 * mm
    pw = (W - lab - 4 * g) / 5
    H = head + sum(pw / bl["asp"] + g for _, _, bl in blocks)
    fig = plt.figure(figsize=(W, H), facecolor="white")
    titles = ["Held-out image", "Reference", "PCGS (this work)", "PGSR", "Asteroid-GS"]
    y = H - head
    for bi, (b, name, bl) in enumerate(blocks):
        hf = pw / bl["asp"]
        for j in range(5):
            ax = fig.add_axes([(lab + j * (pw + g)) / W, (y - hf) / H, pw / W, hf / H])
            if j == 0:
                ax.imshow(bl["image"], cmap="gray", vmin=0, vmax=1, interpolation="lanczos", aspect="auto")
            else:
                c = bl["clay"][KEYS[j - 1]]
                a = (c < 0.999).any(-1, keepdims=True).astype(np.float32)   # white background -> black
                ax.imshow(np.clip(c * a, 0, 1), interpolation="lanczos", aspect="auto")
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if bi == 0:
                ax.set_title(titles[j], fontsize=7, pad=2, fontweight="bold" if j == 2 else "normal")
        fig.text(0.5 * lab / W, (y - hf / 2) / H, name, rotation=90, ha="center", va="center", fontsize=8,
                 fontweight="bold")
        y -= hf + g
    fig.savefig(out + ".png", dpi=300, facecolor="white")
    fig.savefig(out + ".jpg", dpi=200, facecolor="white", pil_kwargs=dict(quality=90))
    print("wrote", out)


# ---------------------------------------------------------------- Fig. learned photometric parameter
def fig_lambert(d, A, out):
    L = json.load(open(f"{d}/lunar_lambert_L.json"))
    ph = json.load(open(f"{d}/phases.json"))
    fig = plt.figure(figsize=(COL, 52 * mm))
    ax = fig.add_axes([0.14, 0.2, 0.83, 0.75])
    mk = {"Ryugu": "o", "Itokawa": "s", "Bennu": "^"}
    ry = [(float(np.mean(ph[k])), float(np.mean(L[k]))) for k in ("ryugu_lp", "ryugu", "ryugu_b2")]
    ax.plot(*zip(*ry), color=C["pcgs"], lw=0.6, alpha=0.6, zorder=2)
    pos = {"ryugu_lp": (1.2, 0.05, "left"), "itokawa": (-1.4, 0.04, "right"), "ryugu": (1.2, -0.06, "left"),
           "ryugu_b2": (1.2, 0.05, "left"), "bennu": (0.0, 0.09, "center")}
    for key, body, _ in CAMPAIGNS:
        p = np.array(ph[key])
        x = float(np.mean(p))
        v = L[key]
        if np.ptp(p) > 5:
            ax.errorbar(x, np.mean(v), xerr=[[x - p.min()], [p.max() - x]], fmt="none", ecolor="#9a9a9a", lw=0.7,
                        capsize=1.8, zorder=1)
        ax.scatter(np.full(len(v), x), v, s=16, marker=mk[body], color=C["pcgs"], edgecolor="white", lw=0.4,
                   zorder=3)
        lab = f"{body} {x:.0f}°" if key != "bennu" else "Bennu 26–60°"
        dx, dy, ha = pos[key]
        ax.text(x + dx, np.mean(v) + dy, lab, fontsize=6.8, ha=ha, va="center", color=INK2)
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Phase angle (°)")
    ax.set_ylabel("Learned partition parameter $L$")
    save(fig, out)


if __name__ == "__main__":
    style()
    kind, d, am, out = sys.argv[1:5]
    A = json.load(open(am))
    dict(campaigns=fig_campaigns, heldout=fig_heldout, overview=fig_overview, lambert=fig_lambert, shapes=fig_shapes, errors=fig_errors, teaser=fig_teaser, night=fig_night, ryugu_phase=fig_ryugu_phase)[kind](d, A, out)
