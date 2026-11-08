"""Supplementary Fig. 1 (Nature Communications layout, 183 mm): accuracy, completeness and where the
error is (the numbers of every metric are in Fig. 3, benchmark/figures/nc_fig_numbers.py).

a-d  share of the reconstructed surface within d of the reference (accuracy) and of the reference
     within d of the reconstruction (completeness), Itokawa and Ryugu; mean over splits, band = split
     range for PCGS; values at d = 1 m for PCGS and the strongest baseline
e-f  mean completeness distance per 5 deg latitude band
Protocol of the main table (dist_profiles.py, collect_metrics.py).
usage: python nc_fig_quant.py <profiles dir> <all_metrics.json> <out>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import result_figs as rf  # noqa: E402

mm = 1 / 25.4
INK, INK2, GRID = rf.INK, rf.INK2, rf.GRID


def frame(ax, grid=None):
    """Axes lines and ticks, no background grid (Nature figure guide)."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=1.5, labelsize=5.6, colors=INK)


def main():
    P = rf.load_profiles(sys.argv[1])
    out = sys.argv[3]
    rf.style()
    plt.rcParams.update({"font.size": 6.2, "axes.labelsize": 6.0})
    W = rf.WMM * mm
    lm, gap = 9.0 * mm, 7.6 * mm
    h1, h2 = 31 * mm, 25 * mm
    H = 5.0 * mm + 7.6 * mm + h1 + 12.5 * mm + h2 + 8.0 * mm
    fig = plt.figure(figsize=(W, H))

    def ax_at(x, y_top, w, h):
        return fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])

    def letter(x, y, s, title=None):
        fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", va="baseline", ha="left", color=INK)
        if title:
            fig.text((x + 3.4 * mm) / W, y / H, title, fontsize=6.5, va="baseline", ha="left", color=INK)

    # legend: one row for the whole figure
    handles = [Line2D([], [], color=c, lw=lw + 0.2, ls=rf.DASH.get(t, "-"), label=l) for t, l, c, lw, _ in rf.M]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False,
               fontsize=6, handlelength=2.0, columnspacing=1.8)

    # ---- a-d: accuracy / completeness curves, grouped by body
    y = H - 5.0 * mm - 7.6 * mm
    w1 = (W - lm - 3 * gap - 1.0 * mm) / 4
    xs = [lm + k * (w1 + gap) for k in range(4)]
    for bi, (body, bname) in enumerate(rf.BODIES):
        xa, xb = xs[2 * bi], xs[2 * bi + 1] + w1
        fig.text((xa + xb) / 2 / W, (y + 6.0 * mm) / H, bname, ha="center", va="center", fontsize=7,
                 fontweight="bold", color=INK)
        fig.add_artist(Line2D([(xa - 6.4 * mm) / W, xb / W], [(y + 4.6 * mm) / H] * 2, color=INK2, lw=0.5))
    letters = iter("abcdef")
    for k, (body, key, title) in enumerate((("itokawa", "acc", "Accuracy"), ("itokawa", "comp", "Completeness"),
                                            ("ryugu", "acc", "Accuracy"), ("ryugu", "comp", "Completeness"))):
        ax = ax_at(xs[k], y, w1, h1)
        frame(ax)
        at1 = {}
        for t, lab, c, lw, z in reversed(rf.M):
            runs = P.get(body, {}).get(t, {})
            if not runs:
                continue
            cv = [rf.cdf(r[key], 3.0) for r in runs.values()]
            x = cv[0][0]
            ys = np.array([v[1] for v in cv])
            if t == "pcgs-v18n" and len(ys) > 1:
                ax.fill_between(x, ys.min(0), ys.max(0), color=c, alpha=0.22, lw=0, zorder=z - 0.5)
            ax.plot(x, ys.mean(0), color=c, lw=lw, ls=rf.DASH.get(t, "-"), zorder=z, solid_capstyle="round")
            at1[t] = ys.mean(0)[np.searchsorted(x, 1.0)]
        ax.axvline(1, color="#bdbdbd", lw=0.5, ls=(0, (2, 1.5)), zorder=1)
        rival = max((t for t in at1 if not t.startswith("pcgs")), key=lambda t: at1[t])
        ax.text(0.97, 0.30, "Within 1 m", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=INK)
        for n, t in enumerate(("pcgs-v18n", rival)):
            ax.scatter([1], [at1[t]], s=8, color=rf.COLOR[t], edgecolor="white", lw=0.4, zorder=9)
            yb = 0.21 - 0.105 * n
            ax.text(0.86, yb, f"{rf.LABEL[t].replace(' (this work)', '')}  {at1[t]:.1f}%",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=INK,
                    fontweight="bold" if t == "pcgs-v18n" else "normal",
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
            ax.plot([0.89, 0.97], [yb + 0.035] * 2, transform=ax.transAxes, color=rf.COLOR[t], lw=1.4,
                    solid_capstyle="butt", zorder=9)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Distance d (m)", labelpad=1)
        ax.set_ylabel("Surface within d (%)" if k == 0 else "", labelpad=1)
        ax.text(0.04, 0.99, title, transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=INK)
        letter(xs[k] - 6.4 * mm, y + 1.0 * mm, next(letters))

    # ---- e-f: latitude profiles
    y2 = y - h1 - 12.5 * mm
    w2 = (W - lm - gap - 1.0 * mm) / 2
    for k, (body, bname) in enumerate(rf.BODIES):
        x0 = lm + k * (w2 + gap)
        ax = ax_at(x0, y2, w2, h2)
        frame(ax, grid="y")
        for a0, a1 in ((-90, -60), (60, 90)):
            ax.axvspan(a0, a1, color="#f4f4f4", lw=0, zorder=0)
        top = 0
        for t, lab, c, lw, z in reversed(rf.M[:4]):
            runs = P.get(body, {}).get(t, {})
            if not runs:
                continue
            e = np.array(next(iter(runs.values()))["lat"]["edges"], float)
            mid = 0.5 * (e[1:] + e[:-1])
            yv = np.nanmean([[np.nan if v is None else v for v in r["lat"]["mean"]] for r in runs.values()], 0)
            ax.plot(mid, yv, color=c, lw=lw, ls=rf.DASH.get(t, "-"), zorder=z)
            top = max(top, np.nanmax(yv))
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, top * 1.1)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xlabel("Latitude on the reference (°)", labelpad=1)
        ax.set_ylabel("Mean distance (m)" if k == 0 else "", labelpad=1)
        ax.text(-75, top * 1.04, "Polar", ha="center", va="top", fontsize=5.3, color=INK2)
        ax.text(75, top * 1.04, "Polar", ha="center", va="top", fontsize=5.3, color=INK2)
        letter(x0 - 6.4 * mm, y2 + 2.0 * mm, next(letters), f"{bname}: completeness distance by latitude")

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600)
    print("wrote", out, f"{H / mm:.0f} mm high")


if __name__ == "__main__":
    main()
