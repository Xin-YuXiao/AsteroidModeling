"""Paper figures for the image track in Nature style (SciencePlots ["science", "nature",
"no-latex"] + Liberation Sans, metric-compatible with Arial: 5-7 pt text, bold 8 pt lowercase
panel letters, 183 mm double-column width, editable PDF text). CPU only
(benchmark/figures/raycast.py, Open3D/Embree), so they run on any node. The paper set and its
numbering are in benchmark/figures/make_figures.sh and docs/image/FIGURES.md.

Paper set (layouts of Chen et al. 2025, A&A 696, A212, Figs. 5-7, plus the error maps):
  sixzoom    global shape: PCGS (a) and the reference (b) from six directions, two regions
             magnified at the right (Chen Fig. 5)
  distmap    signed distance to the reference on each method's surface from six directions
             (blue = inside, red = outside), colour bar, histograms and statistics
  imagecmp   a held-out spacecraft image and the images synthesised from the reference,
             Asteroid-GS and PCGS with its camera and Sun (Chen Fig. 7)
  local      boulder-scale detail: three regions of a held-out image on both models, the real
             image and the synthetic images (Chen Fig. 6)
Drafts, superseded by the above: compare, overview, synthetic.

Views are orthographic along the body-fixed axes: "View from +Z" looks down from the +Z side.
usage: $PY benchmark/figures/paper_figs.py <kind> <body> <out.png> [--split 0] [--threads 16]
Writes <out>.png (600 dpi) and <out>.pdf. Meshes: results/img/<body>/<run>/<mesh>.ply.
"""
import argparse
import gc
import glob
import json
import os
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import open3d as o3d
import scienceplots  # noqa: F401  (registers the styles)
import trimesh

import raycast

D = os.environ.get("AM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_RYUGU = dict(name="Ryugu", camera="Hayabusa2 ONC-T image", ref="reference (SfM-MVS, Watanabe et al. 2019)",
              ref_short="SfM-MVS (Watanabe+ 2019)", model="SfM-MVS model", bar=200.0, vmax=3.0)
BODY = {"itokawa": dict(name="Itokawa", camera="Hayabusa AMICA image", ref="reference (SPC, Gaskell 2008)",
                        ref_short="SPC (Gaskell 2008)", model="SPC model", bar=100.0, vmax=3.0),
        "ryugu": _RYUGU,
        "ryugu_b2": _RYUGU}   # held-out 40 deg phase set (2018-08-31), same reference
OURS = "PCGS (this work)"
METHODS = [("2DGS", "2dgs-dtu_s{s}/mesh_obj"), ("GOF", "gof_s{s}/mesh_gof"),
           ("Asteroid-GS", "ags_s{s}/mesh_ags"), ("PGSR", "pgsr_s{s}/mesh_pgsr"),
           (OURS, "pcgs-v10_s{s}/mesh_mvpoisson_fairoff")]
# FIG_METHODS="label=run_s{s}/mesh;..." replaces the method list (distmap then shows all of them, in
# that order), e.g. to compare PCGS variants
if os.environ.get("FIG_METHODS"):
    METHODS = [tuple(x.split("=", 1)) for x in os.environ["FIG_METHODS"].split(";") if x]
VIEWS = {"+X": ((1, 0, 0), (0, 0, 1)), "-X": ((-1, 0, 0), (0, 0, 1)), "+Y": ((0, 1, 0), (0, 0, 1)),
         "-Y": ((0, -1, 0), (0, 0, 1)), "+Z": ((0, 0, 1), (0, 1, 0)), "-Z": ((0, 0, -1), (0, 1, 0))}
SIX = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
W2 = 7.2           # Nature double-column width (in), 183 mm
THREADS = 16
SUFFIX = ""        # "_sfm": every method's mesh after the shared SfM alignment (evaluation/align_to_sfm.py)
RES = 700
CMAP = "RdBu_r"    # blue = inside the reference, red = outside


def style():
    """Nature: Helvetica/Arial-like sans-serif (Liberation Sans, metric-compatible with Arial, from
    benchmark/figures/fonts), 5-7 pt text, bold 8 pt panel letters, 0.5 pt axes, editable PDF text."""
    plt.style.use(["science", "nature", "no-latex"])
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.ttf"))):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.titlepad": 3, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
                         "ytick.minor.width": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 600, "figure.dpi": 150})


def vname(v):
    return v.replace("-", "−")


# ---------------------------------------------------------------- data (one mesh in memory at a time)
def frame_tf(body):
    return np.array(json.load(open(f"{D}/data/img/{body}/scene/frame_to_ref.json"))["matrix"])


def load_ref(body, scene_frame=False):
    """Reference model; metres in the body frame, or km in the scene (camera) frame."""
    m = trimesh.load(f"{D}/data/img/{body}/ref_mesh_km.ply", process=False)
    if scene_frame:
        m.apply_transform(np.linalg.inv(frame_tf(body)))
    else:
        m.apply_scale(1000.0)
    return m


def largest_component(v, f):
    """Vertices and faces of the connected piece with most faces (scipy, ~100 MB for 7M faces;
    trimesh's split() built one Trimesh per floater fragment and took tens of GB)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]]]).astype(np.int32)
    A = sp.coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(len(v), len(v))).tocsr()
    del e
    _, lab = connected_components(A, directed=False)
    del A
    lf = lab[f[:, 0]]
    f = f[lf == np.bincount(lf).argmax()]
    used = np.unique(f)
    remap = np.full(len(v), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return v[used], remap[f]


def load_rec(body, spec, scene_frame=False):
    """Reconstruction as arrays (.vertices, .faces), largest connected component; metres in the
    body frame via the shared frame registration, or km in its own scene frame."""
    m = trimesh.load(f"{D}/results/img/{body}/{spec}{SUFFIX}.ply", force="mesh", process=False)
    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)
    del m
    f = f[np.isfinite(v).all(1)[f].all(1)]
    v, f = largest_component(v, f)
    if not scene_frame:
        T = frame_tf(body)
        v = (v @ T[:3, :3].T + T[:3, 3]) * 1000.0
    return SimpleNamespace(vertices=v, faces=f)


_REF_SCENE = {}


def signed_dist(ref, v):
    """Signed distance (m, + outside) of points v to the reference (scene built once per reference)."""
    if id(ref) not in _REF_SCENE:
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(o3d.core.Tensor(np.asarray(ref.vertices, np.float32)), o3d.core.Tensor(np.asarray(ref.faces, np.uint32)))
        _REF_SCENE[id(ref)] = s
    r = _REF_SCENE[id(ref)].compute_closest_points(o3d.core.Tensor(np.asarray(v, np.float32)), nthreads=THREADS)
    cp, prim = r["points"].numpy(), r["primitive_ids"].numpy()
    return np.linalg.norm(v - cp, axis=1) * np.sign(((v - cp) * ref.face_normals[prim]).sum(1))


def methods(body, split, keep=None):
    """(label, run/mesh) of the methods whose mesh exists for this body and split."""
    return [(label, spec.format(s=split)) for label, spec in METHODS
            if (keep is None or label in keep) and os.path.exists(f"{D}/results/img/{body}/{spec.format(s=split)}{SUFFIX}.ply")]


# ---------------------------------------------------------------- layout helpers
def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def crop_box(covs, pad=6):
    m = np.any(np.stack(covs), 0)
    ys, xs = np.nonzero(m)
    return max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])


def grid_crop(covs, pad=10, pad_bottom_last=0):
    """Crops for a grid of renders sharing one scale: column j keeps the union of its masks
    horizontally, row i the union of its masks vertically. covs[i][j] are coverage masks.
    Returns ([(x0, x1)] per column, [(y0, y1)] per row)."""
    nr, nc = len(covs), len(covs[0])
    H, W = covs[0][0].shape
    cols, rows = [], []
    for j in range(nc):
        xs = np.nonzero(np.any(np.stack([covs[i][j] for i in range(nr)]), (0, 1)))[0]
        cols.append((max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)))
    for i in range(nr):
        ys = np.nonzero(np.any(np.stack([covs[i][j] for j in range(nc)]), (0, 2)))[0]
        extra = pad_bottom_last if i == nr - 1 else 0
        rows.append((max(ys.min() - pad, 0), min(ys.max() + pad + 1 + extra, H)))
    return cols, rows


def pick_regions(sc, half, views=("+Y", "-Y", "+Z", "-Z"), w_frac=0.16, res=240, n=2, max_shadow=1.0, min_facing=0.0):
    """Square windows (view, (cx, cy, w)) with the most relief in the shaded reference (std of
    the rendered intensity, windows fully on the body and at most `max_shadow` of them in cast
    or attached shadow, mean cosine between surface normal and view direction >= min_facing),
    at most one per view."""
    best = []
    for v in views:
        rgb, cov = sc.lit(*VIEWS[v], half, res, ss=1)
        g = rgb[..., 0]
        ys, xs = np.nonzero(cov)
        w = max(int(w_frac * (xs.max() - xs.min())), 8)
        inner = cv2.erode(cov.astype(np.uint8), np.ones((w + 1, w + 1), np.uint8)) > 0   # half a window from the limb
        nrm, _ = sc.ortho(sc.vn, *VIEWS[v], half, res)
        facing = nrm @ (np.asarray(VIEWS[v][0], float) / np.linalg.norm(VIEWS[v][0]))  # 1 = faces the viewer
        cand = None
        for y in range(ys.min(), ys.max() - w, max(w // 4, 1)):
            for x in range(xs.min(), xs.max() - w, max(w // 4, 1)):
                if inner[y:y + w, x:x + w].all() and (g[y:y + w, x:x + w] < 0.30).mean() <= max_shadow \
                        and facing[y:y + w, x:x + w].mean() >= min_facing:
                    s = g[y:y + w, x:x + w].std()
                    if cand is None or s > cand[0]:
                        cand = (s, x, y)
        if cand:
            s, x, y = cand
            px2 = 2 * half / res
            best.append((s, v, ((x + w / 2) * px2 - half, half - (y + w / 2) * px2, w * px2)))
    best.sort(key=lambda b: -b[0])
    return [(v, win) for _, v, win in best[:n]]


def dist_colors(sd, vmax):
    return plt.get_cmap(CMAP)(np.clip(sd / vmax, -1, 1) * 0.5 + 0.5)[:, :3]


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white")
    print("wrote", out, flush=True)


# ---------------------------------------------------------------- figures
def fig_compare(body, out, split):
    """Columns = methods + reference (titled at the top, mean distance underneath); rows = views."""
    info = BODY[body]
    ref = load_ref(body)
    half = 1.05 * float(np.linalg.norm(ref.vertices, axis=1).max())
    rows = ["+Z", "-Y", "+X"]
    cols = methods(body, split) + [("Reference", None)]
    imgs, sub = {}, {}
    for j, (label, spec) in enumerate(cols):
        m = ref if spec is None else load_rec(body, spec)
        if spec is not None:
            sd = signed_dist(ref, np.asarray(m.vertices))
            sub[j] = f"mean {np.abs(sd).mean():.2f} m"
            del sd
        sc = raycast.Scene(m.vertices, m.faces, THREADS)
        for i, v in enumerate(rows):
            imgs[i, j] = sc.lit(*VIEWS[v], half, RES, ambient=0.28)
        del sc, m
        gc.collect()
        print(" rendered", label, flush=True)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(W2, 0.78 * W2 * len(rows) / len(cols) + 0.35),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02))
    for i, v in enumerate(rows):
        y0, y1, x0, x1 = crop_box([imgs[i, j][1] for j in range(len(cols))])
        for j, (label, _) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(imgs[i, j][0][y0:y1, x0:x1], interpolation="lanczos")
            bare(ax)
            if i == 0:
                ax.set_title(label + ("" if j not in sub else "\n" + sub[j]), fontweight="bold" if label == OURS else "normal")
            if j == 0:
                ax.text(-0.04, 0.5, f"View from {vname(v)}", transform=ax.transAxes, rotation=90, ha="right", va="center")
    axes[0, -1].set_title("Reference\n" + info["model"])
    save(fig, out)


def six_views(sc, half, colors=None, ambient=0.28, res=RES):
    return {v: sc.lit(*VIEWS[v], half, res, colors=colors, ambient=ambient) for v in SIX}


def nice_length(x):
    """A round length (1, 2, 5 x 10^k) close to x."""
    k = 10 ** np.floor(np.log10(x))
    return min((1, 2, 5, 10), key=lambda m: abs(m * k - x)) * k


def fig_overview(body, out, split):
    """a: PCGS from six directions; b: the reference with the same scale and crops; c: the two
    regions with most relief in the reference (numbered boxes in a and b), magnified for both."""
    info = BODY[body]
    ref = load_ref(body)
    rec = load_rec(body, f"pcgs-v10_s{split}/mesh_mvpoisson_fairoff")
    half = 1.05 * float(np.linalg.norm(ref.vertices, axis=1).max())
    s_rec, s_ref = raycast.Scene(rec.vertices, rec.faces, THREADS), raycast.Scene(ref.vertices, ref.faces, THREADS)
    del rec
    regions = pick_regions(s_ref, half)
    colors = ["#1f77b4", "#d62728"]
    layout = [["+X", "+Y", "+Z"], ["-X", "-Y", "-Z"]]
    views = {"PCGS": six_views(s_rec, half), "ref": six_views(s_ref, half)}
    zooms = {(k, n): sc.lit(*VIEWS[v], half, 420, window=win, ambient=0.28)[0]
             for k, sc in (("PCGS", s_rec), ("ref", s_ref)) for n, (v, win) in enumerate(regions)}
    del s_rec, s_ref
    gc.collect()
    covs = [[views[k][v][1] for v in row] for k in ("PCGS", "ref") for row in layout]
    cols, rows = grid_crop(covs, pad=12)
    rows = [(y0, min(y1 + (44 if i % 2 else 0), RES)) for i, (y0, y1) in enumerate(rows)]   # room for the scale bar
    cw, rh = [x1 - x0 for x0, x1 in cols], [y1 - y0 for y0, y1 in rows]
    s = W2 / sum(cw)                                   # inches per rendered pixel
    title_px = 0.2 / s
    hb = [s * (rh[2 * b] + rh[2 * b + 1] + title_px) for b in (0, 1)]
    hc = W2 / 4 + 0.3
    fig = plt.figure(figsize=(W2, sum(hb) + hc))
    outer = fig.add_gridspec(3, 1, height_ratios=hb + [hc], hspace=0.0)
    px = lambda t: (t / (2 * half) + 0.5) * RES  # noqa: E731
    bar_px = info["bar"] / (2 * half) * RES
    for b, (k, title) in enumerate((("PCGS", OURS), ("ref", info["ref"][0].upper() + info["ref"][1:]))):
        sub = outer[b].subgridspec(3, 3, height_ratios=[title_px, rh[2 * b], rh[2 * b + 1]], width_ratios=cw,
                                   wspace=0.0, hspace=0.0)
        tax = fig.add_subplot(sub[0, :])
        tax.axis("off")
        tax.text(0.0, 0.15, "ab"[b], fontsize=8, fontweight="bold", va="bottom", transform=tax.transAxes)
        tax.text(0.022, 0.15, title, fontsize=7, va="bottom", transform=tax.transAxes,
                 fontweight="bold" if k == "PCGS" else "normal")
        for r, row in enumerate(layout):
            y0, y1 = rows[2 * b + r]
            for c, v in enumerate(row):
                x0, x1 = cols[c]
                ax = fig.add_subplot(sub[1 + r, c])
                ax.imshow(views[k][v][0][y0:y1, x0:x1], interpolation="lanczos")
                bare(ax)
                ax.text(0.03, 0.97, vname(v), transform=ax.transAxes, fontsize=6, color="0.35", va="top")
                for n, (rv, (cx, cy, w)) in enumerate(regions):
                    if rv == v:
                        wp = w / (2 * half) * RES
                        X0, Y0 = px(cx - w / 2) - x0, RES - px(cy + w / 2) - y0
                        ax.add_patch(plt.Rectangle((X0, Y0), wp, wp, fill=False, ec=colors[n], lw=0.9))
                        ax.text(X0 + wp + 4, Y0, str(n + 1), color=colors[n], fontsize=6, fontweight="bold", va="top")
                if r == 1 and c == 0:
                    yb = (y1 - y0) - 14
                    ax.plot([10, 10 + bar_px], [yb, yb], color="black", lw=1.6, solid_capstyle="butt")
                    ax.text(10 + bar_px / 2, yb - 6, f"{info['bar']:.0f} m", ha="center", va="bottom", fontsize=6)
    csub = outer[2].subgridspec(2, 4, height_ratios=[0.3 / (W2 / 4) * 420, 420], wspace=0.05, hspace=0.0)
    for n, (v, (cx, cy, w)) in enumerate(regions):
        for c2, k in enumerate(("PCGS", "ref")):
            ax = fig.add_subplot(csub[1, 2 * n + c2])
            ax.imshow(zooms[k, n], interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(colors[n])
                sp.set_linewidth(1.2)
            ax.set_title(f"Region {n + 1}, " + ("PCGS" if k == "PCGS" else "reference"), color=colors[n], pad=2)
            if n == 0 and c2 == 0:
                letter(ax, "c", dx=-0.03, dy=1.06)
                L = nice_length(w / 4)
                lp = L / w * 420
                ax.plot([14, 14 + lp], [400, 400], color="black", lw=1.6, solid_capstyle="butt")
                ax.text(14 + lp / 2, 392, f"{L:g} m", ha="center", va="bottom", fontsize=6)
    save(fig, out)


def fig_distmap(body, out, split, keep=(OURS, "Asteroid-GS", "PGSR", "2DGS")):
    """Rows = methods; six views coloured by signed distance to the reference (same scale and
    crops for every method) + histogram; shared colour bar."""
    info = BODY[body]
    vmax = info["vmax"]
    ref = load_ref(body)
    half = 1.05 * float(np.linalg.norm(ref.vertices, axis=1).max())
    if os.environ.get("FIG_METHODS"):
        keep = tuple(label for label, _ in METHODS)
    blocks = sorted(methods(body, split, keep), key=lambda b: keep.index(b[0]))
    bins = np.linspace(-vmax, vmax, 61)
    data = []
    for label, spec in blocks:                         # one mesh in memory at a time
        m = load_rec(body, spec)
        sd = signed_dist(ref, np.asarray(m.vertices))
        sc = raycast.Scene(m.vertices, m.faces, THREADS)
        views = six_views(sc, half, colors=dist_colors(sd, vmax), ambient=0.55)
        inside = np.abs(sd) <= vmax
        h, _ = np.histogram(sd[inside], bins)          # beyond +-vmax is reported, not piled into end bins
        data.append((label, views, 100 * h / len(sd), np.abs(sd).mean(), np.sqrt((sd ** 2).mean()),
                     100 * (1 - inside.mean())))
        del m, sd, sc
        gc.collect()
        print(" row", label, flush=True)
    # crops from the first row (PCGS) for every row: one scale per column, and the far-flung floaters
    # of some baselines (2DGS) are cut instead of shrinking every body
    cols, rows = grid_crop([[data[0][1][v][1] for v in SIX]], pad=12)
    rows = rows * len(data)
    cw, rh = [x1 - x0 for x0, x1 in cols], [y1 - y0 for y0, y1 in rows]
    hist_w = 0.30 * sum(cw)
    lab_in = 1.20                                      # left column: method name and its statistics (in)
    s = (W2 - lab_in - 0.35) / (sum(cw) + 0.05 * sum(cw) + hist_w)   # inches per pixel
    head, cbar = 0.30 / s, 0.09 / s
    fig = plt.figure(figsize=(W2, s * (sum(rh) + head + cbar) + 0.5))
    gs = fig.add_gridspec(len(data) + 2, 9, width_ratios=[lab_in / s] + cw + [0.05 * sum(cw), hist_w],
                          height_ratios=[head] + rh + [cbar], wspace=0.0, hspace=0.07)
    t = fig.add_subplot(gs[0, 1:7])
    t.axis("off")
    t.text(0.5, 0.95, "Viewing direction (body-fixed frame)", ha="center", va="top", transform=t.transAxes, color="0.3")
    for c, v in enumerate(SIX):                        # short column titles: the view direction
        t = fig.add_subplot(gs[0, 1 + c])
        t.axis("off")
        t.text(0.5, 0.05, vname(v), ha="center", va="bottom", transform=t.transAxes)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for b, (label, views, hp, mean, rmse, beyond) in enumerate(data):
        y0, y1 = rows[b]
        lab = fig.add_subplot(gs[1 + b, 0])
        lab.axis("off")
        lab.text(0.0, 0.86, "abcdefg"[b], fontsize=8, fontweight="bold", va="center", transform=lab.transAxes)
        lab.text(0.13, 0.86, label, va="center", transform=lab.transAxes,
                 fontweight="bold" if label == OURS else "normal")
        lab.text(0.13, 0.70, f"mean |d| {mean:.2f} m\nRMSE {rmse:.2f} m\n|d| > {vmax:g} m: {beyond:.1f}%",
                 va="top", transform=lab.transAxes, fontsize=5.5, color="0.25", linespacing=1.2)
        for c, v in enumerate(SIX):
            x0, x1 = cols[c]
            ax = fig.add_subplot(gs[1 + b, 1 + c])
            ax.imshow(views[v][0][y0:y1, x0:x1], interpolation="lanczos")
            bare(ax)
        ax = fig.add_subplot(gs[1 + b, 8])
        ax.bar(ctr, hp, width=bins[1] - bins[0], color=plt.get_cmap(CMAP)(ctr / vmax * 0.5 + 0.5), lw=0)
        ax.axvline(0, color="0.3", lw=0.4)
        ax.set_xlim(-vmax, vmax)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2, integer=True, prune="upper"))
        ax.yaxis.tick_right()
        ax.tick_params(which="both", top=False, labelsize=6)
        if b == 0:
            ax.set_title("Share of vertices (%)", fontsize=6, loc="right")
        if b == len(data) - 1:
            ax.set_xlabel("Signed distance d (m)")
        else:
            ax.set_xticklabels([])
    cax = fig.add_subplot(gs[-1, 2:6])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vmax, vmax)), cax=cax, orientation="horizontal")
    cb.set_label("Signed distance d to the reference (m)", labelpad=2)
    cb.ax.tick_params(which="both", top=False)
    cax.text(-0.01, 0.5, "inside", transform=cax.transAxes, ha="right", va="center", fontsize=6, color="0.3")
    cax.text(1.01, 0.5, "outside", transform=cax.transAxes, ha="left", va="center", fontsize=6, color="0.3")
    sax = fig.add_subplot(gs[-1, 6])  # scale bar under the last view column
    sax.axis("off")
    sax.set_xlim(0, cw[5])
    sax.set_ylim(0, 1)
    bp = info["bar"] / (2 * half) * RES
    sax.plot([cw[5] - 10 - bp, cw[5] - 10], [0.8, 0.8], color="black", lw=1.6, solid_capstyle="butt")
    sax.text(cw[5] - 10 - bp / 2, 0.5, f"{info['bar']:.0f} m", ha="center", va="top", fontsize=6)
    save(fig, out)


def fig_synthetic(body, out, split, test_index=None, L=0.67):
    """a: a held-out spacecraft image; b-d: images synthesised from three shape models under
    the same camera and Sun (lunar-Lambert, cast shadows), cropped to the body centre."""
    info = BODY[body]
    cams = raycast.load_cameras(f"{D}/data/img/{body}/scene_split{split}", "test")
    cam = cams[test_index if test_index is not None else (4 if body == "itokawa" else 0)]
    panels = [(info["camera"], None),
              ("Synthetic: " + info["model"], lambda: load_ref(body, scene_frame=True)),
              ("Synthetic: Asteroid-GS", lambda: load_rec(body, f"ags_s{split}/mesh_ags", scene_frame=True)),
              ("Synthetic: " + OURS, lambda: load_rec(body, f"pcgs-v10_s{split}/mesh_mvpoisson_fairoff", scene_frame=True))]
    real = cam["img"]
    ys, xs = np.nonzero(cam["mask"])
    side = int(0.62 * max(ys.max() - ys.min(), xs.max() - xs.min()))
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = max(cy - side // 2, 0), max(cx - side // 2, 0)
    win = (slice(y0, y0 + side), slice(x0, x0 + side))
    vmax = np.percentile(real[cam["mask"]], 99.7)
    fig, axes = plt.subplots(2, 2, figsize=(W2 * 0.75, W2 * 0.78), gridspec_kw=dict(wspace=0.04, hspace=0.10))
    for n, (ax, (label, loader)) in enumerate(zip(axes.ravel(), panels)):
        if loader is None:
            img = real
        else:
            m = loader()
            img, fg = raycast.Scene(m.vertices, m.faces, THREADS).camera(cam, L)
            # display stretch, identical for every synthetic panel: match the real image's mean and
            # standard deviation over the body (at 10-20 deg phase the shading alone is nearly flat)
            mk = fg & cam["mask"]
            img = (img - img[mk].mean()) / max(img[mk].std(), 1e-6) * real[mk].std() + real[mk].mean()
            img[~fg] = 0
            del m
            gc.collect()
        ax.imshow(np.clip(img[win] / vmax, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        bare(ax)
        ax.set_title(label, fontweight="bold" if OURS in label else "normal")
        letter(ax, "abcd"[n], dx=-0.01, dy=1.01)
    fig.text(0.5, 0.06, f"{cam['name'].split('.')[0]}, {cam['utc'][:16].replace('T', ' ')} UTC, phase angle {cam['phase']:.1f}° "
             f"(held-out test image)", ha="center", fontsize=6, color="0.3")
    print(f" camera {cam['name']} phase {cam['phase']:.1f} deg", flush=True)
    save(fig, out)


# ---------------------------------------------------------------- after Chen et al. (2025, A&A 696, A212), Figs. 5-7, Nature style
BOX = ["#0072B2", "#E69F00", "#009E73"]   # Okabe-Ito blue, orange, green (colour-blind safe): regions 1-3


def ax_in(fig, x, y, w, h):
    """Bare axes w x h inches whose top left corner is (x, y) inches from the figure's top left."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    bare(ax)
    return ax


def fig_text(fig, x, y, s, **kw):
    """Text with its top left at (x, y) inches from the figure's top left."""
    W, H = fig.get_size_inches()
    fig.text(x / W, 1 - y / H, s, ha=kw.pop("ha", "left"), va=kw.pop("va", "top"), **kw)


def heading(fig, x, y, letter_, title=None):
    """Nature panel letter (bold 8 pt lowercase) and an optional 7 pt heading after it."""
    fig_text(fig, x, y, letter_, fontsize=8, fontweight="bold")
    if title:
        fig_text(fig, x + 0.15, y + 0.012, title, fontsize=7)


def scale_bar(ax, length_px, label, x, y, color="black", outline=None):
    """Bar of length_px data pixels from (x, y) with its label just above (6 pt)."""
    import matplotlib.patheffects as pe
    fx = [pe.withStroke(linewidth=2.2, foreground=outline)] if outline else None
    ax.plot([x, x + length_px], [y, y], color=color, lw=1.2, solid_capstyle="butt", path_effects=fx)
    ax.annotate(label, (x + length_px / 2, y), xytext=(0, 1.5), textcoords="offset points", ha="center",
                va="bottom", fontsize=6, color=color, path_effects=fx)


def tag(ax, s, color, x=0.025, y=0.975):
    """Region number in the panel's top left corner, white on the region colour."""
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=6, fontweight="bold", color="white",
            bbox=dict(fc=color, ec="none", boxstyle="square,pad=0.25"))


def frame(ax, color, lw=1.0):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def fig_sixzoom(body, out, split):
    """Global shape (layout of Chen et al. 2025, Fig. 5). a, PCGS; b, the reference: views from
    +X +Y +Z (upper rows) and -X -Y -Z (lower rows) at one scale; regions 1 and 2 (boxes; the
    most shaded relief in the X/Y views of the reference with little shadow) magnified at right."""
    info = BODY[body]
    layout = [["+X", "+Y", "+Z"], ["-X", "-Y", "-Z"]]
    zres = 560
    views, zooms = {}, {}
    for k in ("ref", "PCGS"):                             # one mesh in memory at a time
        m = load_ref(body) if k == "ref" else load_rec(body, f"pcgs-v10_s{split}/mesh_mvpoisson_fairoff")
        if k == "ref":
            half = 1.05 * float(np.linalg.norm(m.vertices, axis=1).max())
        sc = raycast.Scene(m.vertices, m.faces, THREADS)
        del m
        if k == "ref":   # equatorial views as in Chen et al. (the poles are seen only obliquely), little shadow
            regions = [pick_regions(sc, half, views=tuple(row[:2]), w_frac=0.12, n=1, max_shadow=0.05,
                                    min_facing=0.8)[0] for row in layout]
        views[k] = six_views(sc, half)
        for r, (v, win) in enumerate(regions):
            zooms[k, r] = sc.lit(*VIEWS[v], half, zres, window=win, ambient=0.28)[0]
        del sc
        gc.collect()
        print(" rendered", k, flush=True)
    covs = [[views[k][v][1] for v in row] for k in ("PCGS", "ref") for row in layout]
    cols, rows = grid_crop(covs, pad=10)
    cw, rh = [x1 - x0 for x0, x1 in cols], [y1 - y0 for y0, y1 in rows]
    mg, gap, head, sep, foot = 0.03, 0.14, 0.19, 0.10, 0.17   # margin, views|zooms gap, heading, block gap, scale bar row (in)
    s = (W2 - 2 * mg - gap) / (sum(cw) + min(rh))         # inches per rendered pixel; zoom side = shortest row
    z = s * min(rh)
    H = 2 * mg + 2 * head + sep + s * sum(rh) + foot
    fig = plt.figure(figsize=(W2, H))
    px = lambda t: (t / (2 * half) + 0.5) * RES  # noqa: E731
    titles = (OURS, info["ref"][0].upper() + info["ref"][1:])
    y = mg
    for b, k in enumerate(("PCGS", "ref")):
        heading(fig, mg, y, "ab"[b], titles[b])
        y += head
        for r, row in enumerate(layout):
            y0, y1 = rows[2 * b + r]
            hr = s * (y1 - y0)
            x = mg
            for c, v in enumerate(row):
                x0, x1 = cols[c]
                ax = ax_in(fig, x, y, s * cw[c], hr)
                ax.imshow(views[k][v][0][y0:y1, x0:x1], interpolation="lanczos")
                ax.text(0.02, 0.98, vname(v) + " view", transform=ax.transAxes, fontsize=6, va="top")
                rv, (cx, cy, w) = regions[r]
                if rv == v:
                    wp = w / (2 * half) * RES
                    X0, Y0 = px(cx - w / 2) - x0, RES - px(cy + w / 2) - y0
                    ax.add_patch(plt.Rectangle((X0, Y0), wp, wp, fill=False, ec=BOX[r], lw=0.9))
                    ax.text(X0 + wp + 4, Y0, str(r + 1), color=BOX[r], fontsize=6, fontweight="bold", va="top")
                x += s * cw[c]
            ax = ax_in(fig, W2 - mg - z, y + (hr - z) / 2, z, z)
            ax.imshow(zooms[k, r], interpolation="lanczos")
            frame(ax, BOX[r], 1.2)
            tag(ax, str(r + 1), BOX[r])
            if b == 1:
                w = regions[r][1][2]
                L = nice_length(w / 4)
                scale_bar(ax, L / w * zres, f"{L:g} m", x=14, y=zres - 16)
            y += hr
        y += sep if b == 0 else 0
    sax = ax_in(fig, mg + s * (cw[0] + cw[1]), y + 0.02, s * cw[2], foot - 0.02)   # view scale bar, under the last column
    sax.set_xlim(0, cw[2])
    sax.set_ylim(1, 0)
    bp = info["bar"] / (2 * half) * RES
    scale_bar(sax, bp, f"{info['bar']:.0f} m", x=cw[2] - 10 - bp, y=0.9)
    save(fig, out)


def best_cams(cams, n=2, min_sep=45.0):
    """The n test cameras with the largest body in the frame, preferring viewing directions at
    least min_sep degrees apart."""
    ext = [np.ptp(np.nonzero(c["mask"])[0]) * np.ptp(np.nonzero(c["mask"])[1]) for c in cams]
    order = list(np.argsort(ext)[::-1])
    unit = lambda c: c["C"] / np.linalg.norm(c["C"])  # noqa: E731
    out = []
    for i in order:
        if all(np.degrees(np.arccos(np.clip(unit(cams[i]) @ unit(o), -1, 1))) >= min_sep for o in out):
            out.append(cams[i])
    out += [cams[i] for i in order if not any(cams[i] is o for o in out)]
    return out[:n]


def relief_windows(cam, sizes, erode=0.05):
    """Square windows (x0, y0, w) in image pixels, one per size (fraction of the body extent),
    over the most textured parts of the real image (intensity std), inside the body mask eroded
    by `erode` x the extent (no limb), mutually disjoint."""
    ys, xs = np.nonzero(cam["mask"])
    ext = max(np.ptp(xs), np.ptp(ys))
    k = max(int(erode * ext), 1)
    inner = cv2.erode(cam["mask"].astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    out = []
    for f in sizes:
        best = None
        while best is None and f > 0.05:                  # shrink until a window fits beside the others
            w = int(f * ext)
            for y in range(ys.min(), ys.max() - w, max(w // 6, 1)):
                for x in range(xs.min(), xs.max() - w, max(w // 6, 1)):
                    if any(x < ox + ow and ox < x + w and y < oy + ow and oy < y + w for ox, oy, ow in out):
                        continue
                    if inner[y:y + w, x:x + w].all():
                        sd = cam["img"][y:y + w, x:x + w].std()
                        if best is None or sd > best[0]:
                            best = (sd, x, y)
            f *= 0.8
        if best:
            out.append((best[1], best[2], w))
    return out


def lamp(cam, d=(-0.45, -0.55, -0.70)):
    """Scene-frame direction of a lamp at the upper left of the camera, on the viewer's side."""
    v = cam["R"].T @ np.asarray(d, float)               # camera axes: x right, y down, z forward
    return v / np.linalg.norm(v)


class CamRender:
    """Renders of one mesh (scene frame) through one camera: grey shape under a lamp, and the
    synthetic image (lunar-Lambert, Sun, cast shadows) stretched linearly to the real image's
    mean and std over the body (one stretch per mesh and camera, from the full frame)."""

    def __init__(self, sc, cam, L=0.67):
        self.sc, self.cam, self.L = sc, cam, L
        img, fg = sc.camera(cam, L)
        mk = fg & cam["mask"]
        real = cam["img"][mk]
        self.a = real.std() / max(img[mk].std(), 1e-6)
        self.b = real.mean() - self.a * img[mk].mean()

    def shape(self, window, scale, ambient=0.28):
        img, fg = self.sc.camera(self.cam, 0.0, sun=lamp(self.cam), window=window, scale=scale)
        return np.where(fg, 0.86 * (ambient + (1 - ambient) * img), 1.0)

    def synthetic(self, window, scale):
        img, fg = self.sc.camera(self.cam, self.L, window=window, scale=scale)
        return np.where(fg, self.a * img + self.b, 0.0)


def gray(ax, img, vmax=1.0):
    ax.imshow(np.clip(img / vmax, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")


def gsd(body, sc, cam, win):
    """Metres per image pixel on the surface at the centre of window (x0, y0, w): range along
    the centre ray (scene units -> metres via the frame registration) / focal length."""
    x0, y0, w = win
    d = np.array([(x0 + w / 2 - cam["cx"]) / cam["fx"], (y0 + w / 2 - cam["cy"]) / cam["fy"], 1.0]) @ cam["R"]
    hit, t, _, _ = sc.cast(cam["C"][None], (d / np.linalg.norm(d))[None])
    k = 1000.0 * abs(np.linalg.det(frame_tf(body)[:3, :3])) ** (1 / 3)
    return float(t[0]) * k / cam["fx"] if hit[0] else np.nan


def note(out, text):
    """Caption facts (camera, phase, windows) next to the figure: <out>.txt."""
    with open(os.path.splitext(out)[0] + ".txt", "w") as f:
        f.write(text.strip() + "\n")
    print(text, flush=True)


def fig_local(body, out, split):
    """Boulder-scale detail (layout of Chen et al. 2025, Fig. 6) in the held-out image with the
    largest body. a, PCGS seen by that camera, regions 1-3 boxed (most textured parts of the
    real image). b, region 1 on PCGS and on the reference. c, d, regions 2 and 3: PCGS, the
    reference, the real image, and the images synthesised from PCGS and from the reference."""
    info = BODY[body]
    cam = best_cams(raycast.load_cameras(f"{D}/data/img/{body}/scene_split{split}", "test"), 1)[0]
    wins = relief_windows(cam, (0.28, 0.19, 0.19))
    ys, xs = np.nonzero(cam["mask"])
    pad = 12
    ov = (max(xs.min() - pad, 0), max(ys.min() - pad, 0))
    ov = ov + (min(xs.max() + pad, cam["w"]) - ov[0], min(ys.max() + pad, cam["h"]) - ov[1])
    ov_scale = 900 / max(ov[2], ov[3])
    vmax = np.percentile(cam["img"][cam["mask"]], 99.7)
    R = {}
    for k in ("ref", "PCGS"):                            # one mesh in memory at a time
        m = load_ref(body, scene_frame=True) if k == "ref" else \
            load_rec(body, f"pcgs-v10_s{split}/mesh_mvpoisson_fairoff", scene_frame=True)
        cr = CamRender(raycast.Scene(m.vertices, m.faces, THREADS), cam)
        del m
        if k == "ref":
            mpp = [gsd(body, cr.sc, cam, win) for win in wins]   # metres per image pixel, per window
        for n, (x, y, w) in enumerate(wins):
            R[k, n, "shape"] = cr.shape((x, y, w, w), (760 if n == 0 else 480) / w)
            if n > 0:
                R[k, n, "syn"] = cr.synthetic((x, y, w, w), 480 / w)
        if k == "PCGS":
            R["overview"] = cr.shape(ov, ov_scale)
        del cr
        gc.collect()
        print(" rendered", k, flush=True)
    model = info["model"].replace(" model", "")
    mg, gp, head, rowgap = 0.03, 0.06, 0.19, 0.12       # margin, panel gap, heading, gap between rows (in)
    pb = (W2 * 0.62 - gp) / 2                            # region 1 panels
    ov_w = W2 - 2 * mg - 2 * pb - gp - 0.15
    ov_h = min(ov_w * ov[3] / ov[2], pb)
    ov_w = ov_h * ov[2] / ov[3]
    p = (W2 - 2 * mg - 4 * gp) / 5                       # rows c, d
    H = 2 * mg + head + pb + rowgap + 2 * (head + p) + rowgap
    fig = plt.figure(figsize=(W2, H))
    # a: overview with regions 1-3
    heading(fig, mg, mg, "a", "PCGS shape in the camera view")
    ax = ax_in(fig, mg + 0.05, mg + head, ov_w, ov_h)
    ax.imshow(R["overview"], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    for n, (x, y, w) in enumerate(wins):
        X, Y, Wp = (x - ov[0]) * ov_scale, (y - ov[1]) * ov_scale, w * ov_scale
        ax.add_patch(plt.Rectangle((X, Y), Wp, Wp, fill=False, ec=BOX[n], lw=1.0))
        ax.text(X + Wp + 6, Y, str(n + 1), color=BOX[n], fontsize=6, fontweight="bold", va="top")
    # b: region 1 on both models
    xb = W2 - mg - 2 * pb - gp
    heading(fig, xb - 0.15, mg, "b")
    for c, (k, title) in enumerate((("PCGS", "PCGS shape (this work)"), ("ref", f"Reference shape ({model})"))):
        ax = ax_in(fig, xb + c * (pb + gp), mg + head, pb, pb)
        ax.imshow(R[k, 0, "shape"], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.set_title(title, pad=3)
        frame(ax, BOX[0])
        tag(ax, "1", BOX[0])
        if c == 0:
            w = wins[0][2] * mpp[0]
            L = nice_length(w / 4)
            n_px = R[k, 0, "shape"].shape[0]
            scale_bar(ax, L / w * n_px, f"{L:g} m", x=0.04 * n_px, y=0.95 * n_px)
    # c, d: regions 2 and 3 on models, real image and synthetic images
    titles = ("PCGS shape", "Reference shape", f"Real image ({info['camera'].split()[1]})",
              "Simulated from PCGS", "Simulated from reference")
    y = mg + head + pb + rowgap
    for n in (1, 2):
        heading(fig, mg, y, "cd"[n - 1])
        x, yy, w = wins[n]
        panels = (R["PCGS", n, "shape"], R["ref", n, "shape"], cam["img"][yy:yy + w, x:x + w],
                  R["PCGS", n, "syn"], R["ref", n, "syn"])
        for c, img in enumerate(panels):
            ax = ax_in(fig, mg + c * (p + gp), y + head, p, p)
            if c < 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            else:
                gray(ax, img, vmax)
            ax.set_title(titles[c], pad=3)
            frame(ax, BOX[n])
            if c == 0:
                tag(ax, str(n + 1), BOX[n])
                L = nice_length(w * mpp[n] / 4)
                scale_bar(ax, L / (w * mpp[n]) * img.shape[0], f"{L:g} m", x=0.05 * img.shape[0], y=0.93 * img.shape[0])
        y += head + p + rowgap
    note(out, f"camera {cam['name']} ({cam['utc'][:16].replace('T', ' ')} UTC), phase {cam['phase']:.1f} deg, "
              f"held-out test image of split {split}; windows (x0, y0, w px) {wins}; "
              f"metres per pixel {', '.join(f'{v:.2f}' for v in mpp)}")
    save(fig, out)


def fig_imagecmp(body, out, split, ncam=1):
    """Real vs synthetic images (layout of Chen et al. 2025, Fig. 7). a, a held-out spacecraft
    image; b-d, images synthesised from the reference, Asteroid-GS and PCGS with its camera and
    Sun (lunar-Lambert, cast shadows, one linear stretch to the real image's mean and std over
    the body). One figure per camera for the ncam test images with the largest body."""
    info = BODY[body]
    cams = best_cams(raycast.load_cameras(f"{D}/data/img/{body}/scene_split{split}", "test"), ncam)
    wins = []
    for cam in cams:                                     # square crop shifted towards the terminator
        ys, xs = np.nonzero(cam["mask"])
        ext = max(np.ptp(xs), np.ptp(ys))
        side = int(0.74 * ext)
        sun_img = (cam["R"] @ cam["sun"])[:2]
        sh = -0.16 * ext * sun_img / max(np.linalg.norm(sun_img), 1e-9)
        x0 = int(np.clip(xs.mean() + sh[0] - side / 2, 0, cam["w"] - side))
        y0 = int(np.clip(ys.mean() + sh[1] - side / 2, 0, cam["h"] - side))
        wins.append((x0, y0, side, side))
    models = [(info["ref"], lambda: load_ref(body, scene_frame=True))]
    models += [(label, (lambda sp: lambda: load_rec(body, sp, scene_frame=True))(spec))
               for label, spec in methods(body, split, ("Asteroid-GS", OURS))]
    syn, mpp = {}, {}
    for j, (label, loader) in enumerate(models):         # one mesh in memory at a time
        m = loader()
        sc = raycast.Scene(m.vertices, m.faces, THREADS)
        del m
        for i, (cam, win) in enumerate(zip(cams, wins)):
            syn[label, i] = CamRender(sc, cam).synthetic(win, 2)
            if j == 0:
                mpp[i] = gsd(body, sc, cam, win[:3])
        del sc
        gc.collect()
        print(" rendered", label, flush=True)
    base, ext_ = os.path.splitext(out)
    for i, (cam, (x0, y0, side, _)) in enumerate(zip(cams, wins)):
        o = out if i == 0 else f"{base}_{i + 1}{ext_}"
        vmax = np.percentile(cam["img"][cam["mask"]], 99.7)
        panels = [(f"Real image: {info['camera'].replace(' image', '')} (not used in training)",
                   cam["img"][y0:y0 + side, x0:x0 + side])]
        panels += [("Simulated from " + label, syn[label, i]) for label, _ in models]
        nr = (len(panels) + 1) // 2
        mg, gp, head = 0.03, 0.10, 0.19
        p = (W2 - 2 * mg - gp) / 2
        fig = plt.figure(figsize=(W2, 2 * mg + nr * (head + p) + (nr - 1) * gp))
        for n, (label, img) in enumerate(panels):
            x, y = mg + (n % 2) * (p + gp), mg + (n // 2) * (head + p + gp)
            heading(fig, x, y, "abcd"[n], label)
            ax = ax_in(fig, x, y + head, p, p)
            gray(ax, img, vmax)
            if n == 0:   # black bar on a bright surface, white on the sky
                L = nice_length(side * mpp[i] / 5)
                n_px = img.shape[0]
                lp = L / (side * mpp[i]) * n_px
                under = img[int(0.88 * n_px):, int(0.04 * n_px):int(0.04 * n_px + lp)]
                scale_bar(ax, lp, f"{L:g} m", x=0.04 * n_px, y=0.95 * n_px,
                          color="black" if under.mean() > 0.35 * vmax else "white")
        note(o, f"camera {cam['name']} ({cam['utc'][:16].replace('T', ' ')} UTC), phase {cam['phase']:.1f} deg, "
                f"held-out test image of split {split}; crop (x0, y0, side px) {(x0, y0, side)}, "
                f"{mpp[i]:.2f} m per pixel at the crop centre")
        save(fig, o)


def main():
    global THREADS, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("compare", "overview", "distmap", "synthetic", "sixzoom", "local", "imagecmp"))
    ap.add_argument("body", choices=tuple(BODY))
    ap.add_argument("out")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ncam", type=int, default=1, help="imagecmp: number of cameras (figures)")
    ap.add_argument("--suffix", default="", help="mesh file suffix, e.g. _sfm (SfM-aligned meshes)")
    a = ap.parse_args()
    THREADS, SUFFIX = a.threads, a.suffix
    style()
    kw = dict(ncam=a.ncam) if a.kind == "imagecmp" else {}
    dict(compare=fig_compare, overview=fig_overview, distmap=fig_distmap, synthetic=fig_synthetic,
         sixzoom=fig_sixzoom, local=fig_local, imagecmp=fig_imagecmp)[a.kind](a.body, a.out, a.split, **kw)


if __name__ == "__main__":
    main()
