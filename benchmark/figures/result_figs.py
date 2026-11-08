"""Quantitative result figures (paper 2), Nature style. Every mesh is evaluated with the main-table
protocol: shared SfM alignment + Taubin-20 smoothing, frame registration to the reference,
largest component, 1e6 area-uniform samples (benchmark/figures/dist_profiles.py, evaluation/eval_mesh.py).

  fig_surface    accuracy and completeness curves (share of surface within d) and signed-error density
  fig_metrics    Mean / RMSE / F@1m / normal / PSNR per method, every split shown
  fig_latitude   completeness distance per 5 deg latitude band
  fig_steps      PCGS V10 -> v18 on every campaign and split (dumbbells)

usage: python result_figs.py <profiles dir> <all_metrics.json> <out dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

W2 = 7.2                     # Nature double column, 183 mm
WMM = 183.0                  # figure width (mm); Nature: 89 or 183 mm wide, at most 170 mm high
INK, INK2, GRID = "#000000", "#4d4d4d", "#e6e6e6"
# Nature figure guide: the colour-blind-safe palette of Wong (Nat. Methods 8, 441, 2011), black text only.
# Ours in vermilion, PCGS without priors in orange (dashed), the two strongest baselines in blue and
# bluish green, the rest as grey context.
M = [("pcgs-v18n", "PCGS (this work)", "#D55E00", 1.5, 5),
     ("pcgs-v10", "w/o priors and night-side band", "#E69F00", 0.9, 4),
     ("pgsr", "PGSR", "#0072B2", 1.0, 3),
     ("ags", "Asteroid-GS", "#009E73", 1.0, 3),
     ("2dgs-dtu", "2DGS", "#999999", 0.8, 2),
     ("gof", "GOF", "#C2C2C2", 0.8, 2)]
DASH = {"pcgs-v10": (0, (3, 1.5))}          # dashed as secondary encoding
EXTRA = {"ags-camext": "#80CFB9",            # Asteroid-GS with its repository extent: tint of its colour
         "pcgs-v18": "#EDA67E"}              # PCGS without the night-side band: tint of ours
PRIOR = {"tv": "#CC79A7", "sh": "#56B4E9", "both": "#999999"}          # albedo TV, cast shadows, both
PRIOR_TINT = {"tv": "#F5E4EE", "sh": "#DDEFFA"}
ZOOM = "#F0E442"                             # zoom frames on image tiles


def tile_text(ax, x, y, s, size=6.3, bold=False, color="white", ha="left", va="top", halo=False, box=False, **kw):
    """Text inside an image tile (axes fraction), for black-background panels. No halo by default: a
    path effect turns the text into outlines in the PDF, which the Nature guide does not accept; text that
    crosses a bright surface gets a dark plate (box=True) instead."""
    from matplotlib import patheffects
    pe = [patheffects.withStroke(linewidth=1.2, foreground="black")] if halo and not box else None
    if box:                                    # on bright tiles: a small dark plate instead of a halo
        kw["bbox"] = dict(boxstyle="square,pad=0.22", fc="black", ec="none", alpha=0.6)
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=color, ha=ha, va=va,
                   fontweight="bold" if bold else "normal", path_effects=pe, **kw)


def dark_axes(ax, fg="#d9d9d9"):
    """A plot that sits in a grid of black image tiles."""
    ax.set_facecolor("black")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(fg)
        ax.spines[sp].set_linewidth(0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=fg, labelcolor=fg, length=2, width=0.5, pad=1.2, labelsize=5.5)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
LABEL = {t: l for t, l, *_ in M}
COLOR = dict({t: c for t, _, c, *_ in M}, **EXTRA)
BODIES = [("itokawa", "Itokawa"), ("ryugu", "Ryugu")]


def style():
    try:
        plt.style.use(["science", "nature", "no-latex"])
    except OSError:
        pass
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.ttf"))):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.visible": False, "ytick.minor.visible": False, "xtick.top": False, "ytick.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "pdf.fonttype": 42, "savefig.dpi": 400, "figure.dpi": 150})


def hgrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def letter(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, out):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out + ".png")


def load_profiles(d):
    P = {}
    for f in glob.glob(os.path.join(d, "*__*__s*.json")):
        b, t, s = os.path.basename(f)[:-5].split("__")
        P.setdefault(b, {}).setdefault(t, {})[s[1:]] = json.load(open(f))
    return P


def cdf(h, xmax):
    c = np.cumsum(h["counts"]) + h["under"]
    tot = c[-1] + h["over"]
    x = h["lo"] + h["step"] * np.arange(1, len(c) + 1)
    k = x <= xmax + 1e-9
    return np.r_[0, x[k]], 100 * np.r_[0, c[k]] / tot


def density(h, lo, hi, smooth=3):
    c = np.asarray(h["counts"], float)
    tot = c.sum() + h["under"] + h["over"]
    x = h["lo"] + h["step"] * (np.arange(len(c)) + 0.5)
    if smooth > 1:
        c = np.convolve(c, np.ones(smooth) / smooth, mode="same")
    y = c / tot / h["step"]
    k = (x >= lo) & (x <= hi)
    return x[k], y[k]


def label_column(ax, items, x, ymin, ymax, gap):
    """Direct labels at the right of a set of curves, pushed apart vertically by at least `gap`."""
    items = sorted(items, key=lambda it: it[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - ymax if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    for (y0, txt, c, bold), y in zip(items, ys):
        ax.annotate(txt, (x, y0), xytext=(x + 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y),
                    textcoords="data", va="center", ha="left", fontsize=5.5, color=INK if bold else INK2,
                    fontweight="bold" if bold else "normal",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=0, shrinkB=1), annotation_clip=False)


def fig_surface(P, out, dmax=3.0):
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.15), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for i, (body, bname) in enumerate(BODIES):
        for j, (key, title) in enumerate((("acc", "Accuracy: reconstructed surface within d of the reference"),
                                          ("comp", "Completeness: reference surface within d of the reconstruction"))):
            ax = axes[i, j]
            hgrid(ax, "both")
            at1 = {}
            for t, lab, c, lw, z in reversed(M):
                runs = P.get(body, {}).get(t, {})
                if not runs:
                    continue
                curves = [cdf(r[key], dmax) for r in runs.values()]
                x = curves[0][0]
                ys = np.array([cv[1] for cv in curves])
                if t == "pcgs-v18" and len(ys) > 1:
                    ax.fill_between(x, ys.min(0), ys.max(0), color=c, alpha=0.18, lw=0, zorder=z - 0.5)
                ax.plot(x, ys.mean(0), color=c, lw=lw, zorder=z, solid_capstyle="round", label=lab)
                at1[t] = ys.mean(0)[np.searchsorted(x, 1.0)]
            ax.axvline(1.0, color="#bdbdbd", lw=0.5, zorder=1)
            # selective direct labels: ours and the strongest baseline at d = 1 m
            rival = max((t for t in at1 if not t.startswith("pcgs")), key=lambda t: at1[t])
            for t, dy, va in (("pcgs-v18", 3.5, "bottom"), (rival, -3.5, "top")):
                ax.scatter([1.0], [at1[t]], s=10, color=COLOR[t], edgecolor="white", lw=0.4, zorder=8)
                ax.text(1.06, at1[t] + dy, f"{at1[t]:.1f}%", fontsize=5.6, va=va, ha="left", zorder=9,
                        color=INK if t == "pcgs-v18" else INK2, fontweight="bold" if t == "pcgs-v18" else "normal")
            ax.set_xlim(0, dmax)
            ax.set_ylim(0, 100)
            ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
            ax.set_xlabel("distance d (m)")
            ax.set_ylabel("surface within d (%)" if j == 0 else "")
        ax = axes[i, 2]
        hgrid(ax, "y")
        for t, lab, c, lw, z in reversed(M):
            runs = P.get(body, {}).get(t, {})
            if not runs:
                continue
            xs, ys = zip(*[density(r["signed"], -3, 3) for r in runs.values()])
            y = np.mean(ys, 0)
            if t == "pcgs-v18":
                ax.fill_between(xs[0], 0, y, color=c, alpha=0.15, lw=0, zorder=z - 0.5)
            ax.plot(xs[0], y, color=c, lw=lw, zorder=z)
        ax.axvline(0, color="#bdbdbd", lw=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("signed distance d (m)")
        ax.set_ylabel("probability density (1/m)")
        ax.text(0.02, 0.97, "inside", transform=ax.transAxes, fontsize=5.5, color=INK2, va="top")
        ax.text(0.98, 0.97, "outside", transform=ax.transAxes, fontsize=5.5, color=INK2, va="top", ha="right")
        if i == 0:
            ax.set_title("Signed error of the reconstructed surface", loc="left", fontweight="bold", fontsize=6.5)
        axes[i, 0].text(0.04, 0.96, bname, transform=axes[i, 0].transAxes, va="top", ha="left",
                        fontsize=7.5, fontweight="bold", color=INK)
    axes[0, 0].set_title("Accuracy", loc="left", fontweight="bold", fontsize=6.5)
    axes[0, 1].set_title("Completeness", loc="left", fontweight="bold", fontsize=6.5)
    for ax, l in zip(axes.flat, "abcdef"):
        letter(ax, l)
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = [lab.index(m[1]) for m in M if m[1] in lab]
    fig.tight_layout(w_pad=2.2, h_pad=1.4, rect=(0, 0.05, 1, 1))
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="lower center", ncol=len(order), frameon=False,
               fontsize=6, handlelength=1.8, columnspacing=1.6)
    save(fig, out)


METRICS = [("mean", "Mean distance (m)", False), ("rmse", "RMSE (m)", False), ("f1", "F-score at 1 m", True),
           ("normal", "Normal error (°)", False), ("psnr", "Held-out PSNR (dB)", True)]


def mval(r, k):
    g = r.get("sfm_t20")
    if g is None:
        return None
    if k == "psnr":
        return r.get("nvs", {}).get("matched", {}).get("psnr")
    return {"mean": g["A"]["mean"], "rmse": g["A"]["rmse"], "f1": g["F"]["F@1m"], "normal": g["normal"]["mean"]}[k]


def fig_metrics(A, out):
    rows = [t for t, *_ in M[:4]]           # 2DGS and GOF are off every axis (Table 1)
    fig, axes = plt.subplots(2, len(METRICS), figsize=(W2, 2.55), sharey=True)
    for i, (body, bname) in enumerate(BODIES):
        for j, (k, lab, higher) in enumerate(METRICS):
            ax = axes[i, j]
            hgrid(ax, "x")
            allv, lab_pos = [], []
            for y, t in enumerate(rows):
                v = [mval(r, k) for r in A.get(body, {}).get(t, {}).values()]
                v = [x for x in v if x is not None]
                if not v:
                    continue
                allv += v
                c = COLOR[t]
                ax.plot([min(v), max(v)], [y, y], color=c, lw=0.8, alpha=0.6, solid_capstyle="round", zorder=2)
                ax.scatter(v, [y] * len(v), s=7, facecolor="white", edgecolor=c, lw=0.6, zorder=3)
                ax.scatter([np.mean(v)], [y], s=22 if t == "pcgs-v18" else 14, color=c, edgecolor="white", lw=0.4, zorder=4)
                lab_pos.append((y, max(v), f"{np.mean(v):.3g}" if k != "psnr" else f"{np.mean(v):.1f}", t))
            lo, hi = min(allv), max(allv)
            span = hi - lo
            ax.set_xlim(lo - 0.08 * span, hi + 0.32 * span)
            for y, xr, txt, t in lab_pos:
                ax.text(xr + 0.05 * span, y, txt, ha="left", va="center", fontsize=5.4,
                        color=INK if t == "pcgs-v18" else INK2, fontweight="bold" if t == "pcgs-v18" else "normal")
            ax.set_ylim(len(rows) - 0.4, -0.75)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([LABEL[t] for t in rows])
            ax.tick_params(axis="y", length=0)
            for tl, t in zip(ax.get_yticklabels(), rows):
                tl.set_color(INK if t == "pcgs-v18" else INK2)
                tl.set_fontweight("bold" if t == "pcgs-v18" else "normal")
            ax.spines["left"].set_visible(False)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            if i == 0:
                ax.set_title(lab, fontsize=6.3, color=INK)
            if i == 1:
                ax.set_xlabel("better →" if higher else "← better", fontsize=5.3, color=INK2,
                              loc="right" if higher else "left")
            if j == 0:
                ax.text(-0.72, 0.5, bname, transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=7.5, fontweight="bold", color=INK)
    fig.tight_layout(w_pad=0.9, h_pad=1.1)
    save(fig, out)


def fig_latitude(P, out):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.05), sharey=False)
    for ax, (body, bname), l in zip(axes, BODIES, "ab"):
        hgrid(ax, "y")
        ax.axvspan(-90, -60, color="#f3f3f3", lw=0, zorder=0)
        ax.axvspan(60, 90, color="#f3f3f3", lw=0, zorder=0)
        top = 0
        for t, lab, c, lw, z in reversed(M[:4]):
            runs = P.get(body, {}).get(t, {})
            if not runs:
                continue
            e = np.array(next(iter(runs.values()))["lat"]["edges"], float)
            mid = 0.5 * (e[1:] + e[:-1])
            ys = np.array([[np.nan if v is None else v for v in r["lat"]["mean"]] for r in runs.values()])
            y = np.nanmean(ys, 0)
            ax.plot(mid, y, color=c, lw=lw, zorder=z, solid_joinstyle="round", label=lab)
            top = max(top, np.nanmax(y))
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, top * 1.1)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xlabel("latitude of the reference surface (°)")
        ax.set_ylabel("mean completeness distance (m)")
        ax.set_title(bname, loc="left", fontweight="bold")
        ax.text(-75, top * 1.04, "polar", ha="center", fontsize=5.3, color=INK2)
        ax.text(75, top * 1.04, "polar", ha="center", fontsize=5.3, color=INK2)
        letter(ax, l, x=-0.1)
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(m[1]) for m in M if m[1] in lab]
    fig.tight_layout(w_pad=2.0, rect=(0, 0.08, 1, 1))
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="lower center", ncol=len(order), frameon=False,
               fontsize=6, handlelength=1.8, columnspacing=1.6)
    save(fig, out)


CAMP = [("itokawa", "Itokawa, 10°", "albedo TV"), ("ryugu_lp", "Ryugu, 7°", "albedo TV"),
        ("ryugu", "Ryugu, 18°", "cast shadows"), ("ryugu_b2", "Ryugu, 40°", "cast shadows")]


def fig_steps(A, out):
    rows = []
    for body, name, term in CAMP:
        for s in sorted(set(A.get(body, {}).get("pcgs-v10", {})) & set(A.get(body, {}).get("pcgs-v18", {}))):
            a, b = A[body]["pcgs-v10"][s], A[body]["pcgs-v18"][s]
            if "sfm_t20" in a and "sfm_t20" in b:
                rows.append((f"{name}, split {s}", term, a, b))
    keys = [("rmse", "RMSE (m)", False), ("f1", "F-score at 1 m", True), ("normal", "Normal error (°)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 0.42 + 0.21 * len(rows)), sharey=True)
    for ax, (k, lab, higher), l in zip(axes, keys, "abc"):
        hgrid(ax, "x")
        for y, (name, term, a, b) in enumerate(rows):
            va, vb = mval(a, k), mval(b, k)
            better = (vb > va) if higher else (vb < va)
            ax.plot([va, vb], [y, y], color=COLOR["pcgs-v18"] if better else "#9b9b9b", lw=1.0, zorder=2,
                    solid_capstyle="round")
            ax.scatter([va], [y], s=14, facecolor="white", edgecolor=COLOR["pcgs-v10"], lw=0.9, zorder=3)
            ax.scatter([vb], [y], s=16, color=COLOR["pcgs-v18"], edgecolor="white", lw=0.4, zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r[0]}  ({r[1]})" for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(lab + ("  (higher is better)" if higher else "  (lower is better)"), fontsize=6.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        letter(ax, l, x=-0.05 if l != "a" else -0.62)
    axes[0].scatter([], [], s=14, facecolor="white", edgecolor=COLOR["pcgs-v10"], lw=0.9, label="PCGS V10")
    axes[0].scatter([], [], s=16, color=COLOR["pcgs-v18"], label="PCGS (this work)")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="lower right", frameon=False, fontsize=5.5,
                    handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(w_pad=1.0)
    save(fig, out)


def main():
    P = load_profiles(sys.argv[1])
    A = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    style()
    fig_surface(P, os.path.join(out, "fig_surface"))
    fig_metrics(A, os.path.join(out, "fig_metrics"))
    fig_latitude(P, os.path.join(out, "fig_latitude"))
    fig_steps(A, os.path.join(out, "fig_steps"))


if __name__ == "__main__":
    main()
