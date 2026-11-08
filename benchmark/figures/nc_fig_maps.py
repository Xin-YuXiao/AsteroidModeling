"""Supplementary Fig. 1 (Nature Communications layout, 183 mm): where the shape error lies. Mollweide
(equal-area) maps of the body-fixed reference frame showing the completeness distance, the distance from
area-uniform samples of the reference to the reconstruction, split 0, main-table protocol (err_maps.py),
2 deg bins lightly smoothed (Gaussian, 1 bin, empty bins excluded). Rows PCGS, PGSR, Asteroid-GS; columns
Itokawa, Ryugu; one colour scale.
usage: python nc_fig_maps.py <dir of err_maps npz> <out>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import result_figs as rf  # noqa: E402

mm = 1 / 25.4
INK = rf.INK
ROWS = [("pcgs-v18n", "PCGS (this work)"), ("pgsr", "PGSR"), ("ags", "Asteroid-GS")]
BODIES = [("itokawa", "Itokawa"), ("ryugu", "Ryugu (19°)"), ("bennu", "Bennu")]
TAG = {("bennu", "ags"): "ags-camext"}      # Asteroid-GS ran on Bennu only with its scene extent
VMAX = 2.5
NORM = Normalize(0, VMAX)
CMAP = matplotlib.colormaps["magma_r"].copy()
CMAP.set_bad("#e6e6e6")
CMAP.set_over("#1b0c41")


def smooth(mean, count, sigma=1.0):
    v = np.where(count > 0, mean, 0.0)
    w = (count > 0).astype(float)
    # wrap in longitude, not in latitude
    vs = gaussian_filter(v, sigma, mode=("nearest", "wrap"))
    ws = gaussian_filter(w, sigma, mode=("nearest", "wrap"))
    out = np.where(ws > 0.2, vs / np.maximum(ws, 1e-9), np.nan)
    return np.where(count > 0, out, np.nan)


def main():
    d, out = sys.argv[1:3]
    rf.style()
    W = rf.WMM * mm
    rl, gap, cbw = 6.0 * mm, 4.0 * mm, 15.0 * mm
    mw = (W - rl - 2 * gap - cbw) / 3
    mh = mw / 2
    head, rgap, bottom = 7.0 * mm, 3.0 * mm, 2.0 * mm
    H = head + 3 * mh + 2 * rgap + bottom
    fig = plt.figure(figsize=(W, H), facecolor="white")
    letters = iter("abcdefghi")
    for j, (body, bname) in enumerate(BODIES):
        x = rl + j * (mw + gap)
        fig.text((x + mw / 2) / W, 1 - 3.6 * mm / H, bname, ha="center", va="baseline", fontsize=7, fontweight="bold",
                 color=INK)
        for i, (tag, label) in enumerate(ROWS):
            z = np.load(os.path.join(d, f"{body}__{TAG.get((body, tag), tag)}__s0.npz"))
            le, oe = np.radians(z["lat_edges"]), np.radians(z["lon_edges"])
            lc, oc = 0.5 * (le[1:] + le[:-1]), 0.5 * (oe[1:] + oe[:-1])
            yt = head + i * (mh + rgap)
            ax = fig.add_axes([x / W, 1 - (yt + mh) / H, mw / W, mh / H], projection="mollweide")
            sm = smooth(z["mean"], z["count"], 1.2)
            sm = np.where(np.isnan(sm), np.nanmedian(sm), sm)          # the few empty polar bins
            ax.pcolormesh(oc, lc, sm, cmap=CMAP, norm=NORM, shading="gouraud", rasterized=True)
            ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
            ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color="white", lw=0.35, alpha=0.55)
            ax.spines["geo"].set_linewidth(0.5)
            ax.spines["geo"].set_color(INK)
            c = z["count"]
            m = float(np.nansum(z["mean"] * c) / c.sum())
            ax.text(1.0, 0.02, f"mean {m:.2f} m", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
                    color=INK, fontweight="bold" if tag == "pcgs-v18n" else "normal")
            fig.text((x + 0.6 * mm) / W, 1 - (yt + 0.6 * mm) / H, next(letters), fontsize=8, fontweight="bold",
                     va="top", ha="left", color=INK)
            if j == 0:
                fig.text(1.0 * mm / W, 1 - (yt + mh / 2) / H, label, rotation=90, ha="left", va="center",
                         fontsize=6.5, fontweight="bold" if tag == "pcgs-v18n" else "normal", color=INK)
    cax = fig.add_axes([(W - cbw + 4.0 * mm) / W, 1 - (head + 2.25 * mh + 2 * rgap) / H, 1.8 * mm / W,
                        (1.5 * mh) / H])
    cb = fig.colorbar(ScalarMappable(NORM, CMAP), cax=cax, orientation="vertical", extend="max")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
    cb.set_label("Completeness distance (m)", fontsize=6, labelpad=2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
    print("wrote", out, f"{W / mm:.0f} x {H / mm:.0f} mm")


if __name__ == "__main__":
    main()
