"""Fig. 3, quantitative results (Nature Communications layout, 183 mm): numbers only, no renders.

a, b  Itokawa and Ryugu: every method of Table 1, five metrics; bar = mean over the three splits,
      whisker = s.d., dots = the splits, value at the bar end (bold: best in the panel)
c     Bennu (OLA-based reference), mean over the three splits (PCGS without the night-side band: split 0)
d     (PANEL_D = True only) the phase-dependent priors on split 0 of every campaign: change against
      PCGS without priors; the paper shows these runs in Fig. 9 and Table A.3 instead
Same values as benchmark/paper_tables.py (it is imported for the data access), protocol of Table 1.
Also writes <out>_source_data.csv (every plotted value, per split).
usage: python nc_fig_numbers.py <all_metrics.json> <out>
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.transforms  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import result_figs as rf  # noqa: E402
import paper_tables as pt  # noqa: E402

mm = 1 / 25.4
PANEL_D = False          # Fig. 3 of the paper: final-model results only (a-c)
INK, INK2 = rf.INK, rf.INK2
C = rf.COLOR
MAIN = [("2DGS", "2dgs-dtu"), ("GOF", "gof"), ("PGSR", "pgsr"), ("Asteroid-GS", "ags"),
        ("Asteroid-GS†", "ags-camext"), ("PCGS (this work)", "pcgs-v18n")]
BENNU = [("2DGS", "2dgs-dtu"), ("PGSR", "pgsr"), ("Asteroid-GS†", "ags-camext"), ("w/o night-side band‡", "pcgs-v18"),
         ("PCGS (this work)", "pcgs-v18n")]
# metric, column title, axis limit for a-b, for c (bars beyond it are cut and carry their value)
COLS = [("mean", "Mean distance (m)", 3.0, 6.0), ("rmse", "RMSE (m)", 5.0, 12.0), ("f1", "F-score at 1 m", 1.0, 1.0),
        ("normal", "Normal error (°)", 45.0, 60.0), ("psnr", "Held-out PSNR (dB)", 42.0, 25.0)]
PRIOR_C = rf.PRIOR
PRIORS = [  # campaign, body, (label, run, prior, selected by the phase rule)
    ("Itokawa 10°", "itokawa", [("albedo TV", "pcgs-v14t", "tv", True)]),
    ("Ryugu 7°", "ryugu_lp", [("albedo TV", "pcgs-v18", "tv", True)]),
    ("Ryugu 19°", "ryugu", [("albedo TV", "pcgs-v14t", "tv", False), ("cast shadows", "pcgs-v15s", "sh", True),
                                 ("both", "pcgs-v17", "both", False)]),
    ("Ryugu 40°", "ryugu_b2", [("cast shadows", "pcgs-v15s", "sh", True), ("both", "pcgs-v17", "both", False)]),
    ("Bennu 26–60°", "bennu", [("cast shadows", "pcgs-v18", "sh", True)])]
DCOLS = [("rmse", "Change in RMSE (%) ↓", "rel"), ("f1", "Change in F-score at 1 m ↑", "abs"),
         ("normal", "Change in normal error (°) ↓", "abs"),
         ("psnr", "Change in held-out PSNR (dB) ↑", "abs")]
MINUS = "−"


def fmt(v, nd, sign=False):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def frame(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=2, width=0.5, pad=1.2, labelsize=5.5, colors=INK2)
    ax.tick_params(axis="y", length=0, pad=2, labelsize=6, colors=INK)


def bars(ax, A, body, rows, key, clip, src, splits=None):
    """Horizontal bars, top to bottom in the order of rows; returns nothing, draws everything."""
    nd = pt.METRIC[key][3]
    sign = pt.METRIC[key][2]
    stats = []
    for label, tag in rows:
        per = [(s, pt.values(A, pt.row(label, body, tag, splits=[s]), pt.col(key)))
               for s in (sorted(A.get(body, {}).get(tag, {})) if splits is None else splits)]
        src += [(body, label, key, s, x[0]) for s, x in per if x]
        stats.append([x[0] for _, x in per if x])
    means = [round(sum(v) / len(v), nd) if v else None for v in stats]
    cand = [m for m in means if m is not None]
    best = (max(cand) if sign > 0 else min(cand)) if cand else None
    n = len(rows)
    for i, ((label, tag), v) in enumerate(zip(rows, stats)):
        if not v:
            continue
        y = n - 1 - i
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        cut = m > clip
        ax.barh(y, min(m, clip), height=0.66, color=C[tag], lw=0, zorder=2)
        if cut:   # break mark: the bar continues beyond the axis
            for dx in (0.86, 0.90):
                ax.plot([clip * dx - 0.012 * clip, clip * dx + 0.012 * clip], [y - 0.42, y + 0.42], color="white",
                        lw=1.1, zorder=3, solid_capstyle="butt")
        else:
            if sd > 0:
                ax.errorbar(m, y, xerr=sd, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5,
                            zorder=3)
            for k, x in enumerate(v):
                if x <= clip:
                    ax.scatter([x], [y + (k - (len(v) - 1) / 2) * 0.17], s=2.0, color="white", edgecolor=INK,
                               lw=0.35, zorder=4)
        xr = clip if cut else max([m + sd] + [x for x in v if x <= clip])
        ax.text(xr + 0.035 * clip, y, fmt(m, nd), fontsize=5.2, va="center", ha="left", color=INK,
                fontweight="bold" if means[i] == best else "normal", zorder=5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, clip * 1.36)
    ticks = {3.0: [0, 1, 2, 3], 5.0: [0, 2.5, 5], 1.0: [0, 0.5, 1], 45.0: [0, 20, 40], 42.0: [0, 20, 40],
             6.0: [0, 2, 4, 6], 12.0: [0, 6, 12], 60.0: [0, 30, 60], 25.0: [0, 10, 20]}[clip]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.spines["bottom"].set_bounds(0, clip)


def save(fig, out, src, W, H):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    with open(f"{out}_source_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["body", "method", "metric", "split", "value"])
        w.writerows(src)
    print("wrote", out + ".png", f"{W / mm:.0f} x {H / mm:.0f} mm,", len(src), "source values")


def main():
    A = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    rf.style()
    W = rf.WMM * mm
    lm, rm, gap = 25.5 * mm, 1.0 * mm, 4.2 * mm
    pw = (W - lm - rm - 4 * gap) / 5
    pitch = 2.95 * mm
    hd = 9 * 2.95 * mm + 4 * 0.55 * 2.95 * mm
    tops = {}
    y = 7.5 * mm                               # from the top: column titles, then a, b, c, d
    tops["a"] = y + 4.2 * mm
    tops["b"] = tops["a"] + 6 * pitch + 9.0 * mm
    tops["c"] = tops["b"] + 6 * pitch + 14.5 * mm
    tops["d"] = tops["c"] + 5 * pitch + 16.5 * mm
    H = tops["d"] + hd + 9.5 * mm if PANEL_D else tops["c"] + 5 * pitch + 10.0 * mm
    fig = plt.figure(figsize=(W, H))

    def ax_at(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    def heading(letter, y_top, text):
        fig.text(1.2 * mm / W, 1 - (y_top - 3.2 * mm) / H, letter, fontsize=8, fontweight="bold", va="baseline",
                 color=INK)
        fig.text(5.0 * mm / W, 1 - (y_top - 3.2 * mm) / H, text, fontsize=6.5, va="baseline", color=INK)

    src = []
    # column titles for a-b (the x axes of a and b share these limits)
    for j, (_, title, _, _) in enumerate(COLS):
        arrow = "↑" if pt.METRIC[COLS[j][0]][2] > 0 else "↓"
        fig.text((lm + j * (pw + gap)) / W, 1 - 4.6 * mm / H, f"{title} {arrow}", fontsize=6.3, va="baseline",
                 ha="left", color=INK, fontweight="bold")
    for letter, body, text in (("a", "itokawa", "Itokawa  ·  Hayabusa AMICA, phase 10°  ·  "
                                                "3 splits of 50 training and 10 held-out images"),
                               ("b", "ryugu", "Ryugu  ·  Hayabusa2 ONC-T, phase 19°  ·  3 splits")):
        heading(letter, tops[letter], text)
        for j, (key, title, clip, _) in enumerate(COLS):
            ax = ax_at(lm + j * (pw + gap), tops[letter], pw, 6 * pitch)
            frame(ax)
            bars(ax, A, body, MAIN, key, clip, src)
            if j == 0:
                ax.set_yticks(range(len(MAIN)))
                ax.set_yticklabels([lab for lab, _ in MAIN][::-1])
                for tl in ax.get_yticklabels():
                    tl.set_fontweight("bold" if tl.get_text().startswith("PCGS") else "normal")
            else:
                ax.set_yticks([])
            if letter == "b":
                ax.set_xlabel(title, fontsize=6, labelpad=1.5, color=INK)
    heading("c", tops["c"], "Bennu  ·  OSIRIS-REx MapCam, phase 26–60°  ·  reference from the laser "
                            "altimeter (OLA), independent of the images  ·  three splits (‡ split 0)")
    for j, (key, title, _, clip) in enumerate(COLS):
        ax = ax_at(lm + j * (pw + gap), tops["c"], pw, 5 * pitch)
        frame(ax)
        bars(ax, A, "bennu", BENNU, key, clip, src)
        if j == 0:
            ax.set_yticks(range(len(BENNU)))
            ax.set_yticklabels([lab for lab, _ in BENNU][::-1])
            for tl in ax.get_yticklabels():
                tl.set_fontweight("bold" if tl.get_text().startswith("PCGS") else "normal")
        else:
            ax.set_yticks([])
        ax.set_xlabel(title, fontsize=6, labelpad=1.5, color=INK)

    if not PANEL_D:
        return save(fig, out, src, W, H)
    # d: priors, change against PCGS without priors (split 0)
    heading("d", tops["d"], "Phase rule (development runs without the night-side band), split 0: change against no prior "
                            "(bold: the variant the phase rule selects)")
    ys, labels, groups, yy = [], [], [], 0.0
    for camp, body, variants in PRIORS:
        groups.append((camp, yy))
        for lab, run, prior, sel in variants:
            ys.append((yy, body, run, prior, sel))
            labels.append((yy, lab, sel))
            yy += 1
        yy += 0.55
    span = yy - 0.55
    lmd = 37.0 * mm
    pwd = (W - lmd - rm - 3 * gap) / 4
    for j, (key, title, mode) in enumerate(DCOLS):
        ax = ax_at(lmd + j * (pwd + gap), tops["d"], pwd, hd)
        frame(ax)
        nd = {"rmse": 1, "f1": 3, "normal": 1, "psnr": 1}[key]
        vals = []
        for y0, body, run, prior, sel in ys:
            b = pt.values(A, pt.row("", body, "pcgs-v10", splits=["0"]), pt.col(key))
            v = pt.values(A, pt.row("", body, run, splits=["0"]), pt.col(key))
            if not b or not v:
                vals.append(None)
                continue
            d = 100 * (v[0] / b[0] - 1) if mode == "rel" else v[0] - b[0]
            src.append((body, f"{run} vs pcgs-v10", f"delta_{key}" + ("_pct" if mode == "rel" else ""), "0", d))
            vals.append(d)
        lo, hi = min([0.0] + [x for x in vals if x is not None]), max([0.0] + [x for x in vals if x is not None])
        lim = hi - lo
        for (y0, body, run, prior, sel), d in zip(ys, vals):
            if d is None:
                continue
            ax.barh(y0, d, height=0.66, color=PRIOR_C[prior], lw=0, zorder=2)
            ax.text(d + (0.025 if d >= 0 else -0.025) * lim, y0,
                    fmt(d, nd, sign=True) + ("%" if mode == "rel" else ""), fontsize=5.2, va="center",
                    ha="left" if d >= 0 else "right", color=INK, fontweight="bold" if sel else "normal")
        ax.axvline(0, color=INK, lw=0.5, zorder=3)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(lo - 0.42 * lim, hi + 0.42 * lim)
        ax.set_ylim(span - 0.4, -0.6)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace("-", MINUS)))
        ax.set_xlabel(title, fontsize=6, labelpad=1.5, color=INK)
        if j == 0:
            tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
            for y0, lab, sel in labels:
                ax.text((lmd - 1.0 * mm) / W, y0, lab, fontsize=6, va="center", ha="right", color=INK,
                        fontweight="bold" if sel else "normal", transform=tr)
            for camp, y0 in groups:
                ax.text(1.5 * mm / W, y0, camp, fontsize=6, va="center", ha="left", color=INK, style="italic",
                        transform=tr)
    # legend of d (keylines, black text)
    hl = [Line2D([], [], color=PRIOR_C[k], lw=3.2, label=l) for k, l in
          (("tv", "albedo TV (below 15°)"), ("sh", "cast shadows (15° and above)"), ("both", "both"))]
    fig.legend(handles=hl, loc="lower left", bbox_to_anchor=(lmd / W, 0.0), ncol=3, frameon=False, fontsize=5.8,
               handlelength=1.4, columnspacing=1.6, borderaxespad=0.1)

    save(fig, out, src, W, H)


if __name__ == "__main__":
    main()
