"""Fig. 5 (Nature Communications layout, 183 mm): illumination and the two photometric priors.

a    one held-out image per campaign, ordered by phase angle (Ryugu 7, Itokawa 10, Ryugu 19 and 40 deg)
b-d  PCGS without priors (V10) -> this work on every campaign and split (RMSE, F-score at 1 m, normal error); the prior
     active at that phase is named (albedo TV below 15 deg, cast shadows at and above)
e-h  signed distance of V10 and of this work on Itokawa (albedo TV) and Ryugu (cast shadows), same
     cameras as Fig. 2
usage: python nc_fig_steps.py render <panel dir>                               
       python nc_fig_steps.py compose <panel dir> <all_metrics.json> <out>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CAMPAIGNS = [("ryugu_lp", "Ryugu", 7), ("itokawa", "Itokawa", 10), ("ryugu", "Ryugu", 19), ("ryugu_b2", "Ryugu", 40)]
VIEWS = {"itokawa": "ST_2421076883_v", "ryugu": "hyb2_onc_20180710_060508_tvf_l2c"}
VMAX = 3.0


def render(pdir):
    import paper_figs as pf
    import raycast
    import render_hq as hq
    import nc_fig_recon as rc
    pf.SUFFIX, pf.THREADS = "_sfm_t20", 12
    hq.CLAY[:] = (0.60, 0.585, 0.56)
    os.makedirs(pdir, exist_ok=True)
    meta = {}
    for body, name in VIEWS.items():
        d, up, half, rot, cam, ref = rc.camera_view(pf, raycast, body, name)
        meta[body] = dict(half=half, res=1500, rot=rot, stats={})
        for tag in ("pcgs-v10", "pcgs-v18"):
            m = pf.load_rec(body, f"{tag}_s0/mesh_mvpoisson_fairoff")
            sd = pf.signed_dist(ref, m.vertices)
            r = hq.Renderer(m.vertices, m.faces, 12)
            img = r.render(d, up, half, 1500, colors=hq.diverging(sd, VMAX),
                           **rc.ERR_LIGHT)
            np.save(f"{pdir}/{body}_{tag}_err.npy", img.astype(np.float16))
            meta[body]["stats"][tag] = dict(mean_abs=float(np.abs(sd).mean()), rmse=float(np.sqrt((sd ** 2).mean())),
                                            over3=float((np.abs(sd) > VMAX).mean()))
            print(body, tag, flush=True)
            del r, m
    for body, _, _ in CAMPAIGNS:                     # one held-out image per campaign
        root = pf.D + f"/data/img/{body}"
        cam = raycast.load_cameras(f"{root}/scene_split0", "test")[0]
        ys, xs = np.nonzero(cam["mask"])
        c, s = ((ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2), int(0.56 * max(ys.ptp(), xs.ptp()))
        im = np.pad((cam["img"] * 255).astype(np.uint8), s)[c[0]:c[0] + 2 * s, c[1]:c[1] + 2 * s]
        np.save(f"{pdir}/{body}_thumb.npy", im)
        meta.setdefault("thumbs", {})[body] = dict(name=cam["name"], phase=cam["phase"], utc=cam["utc"])
    json.dump(meta, open(f"{pdir}/meta.json", "w"), indent=1)


PRIOR = None                       # filled in compose() from result_figs (Wong palette)


def compose(pdir, metrics, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    import result_figs as rf
    rf.style()
    INK = rf.INK
    global PRIOR
    PRIOR = {"tv": ("Albedo TV", rf.PRIOR["tv"], rf.PRIOR_TINT["tv"]),
             "shadow": ("Cast shadows", rf.PRIOR["sh"], rf.PRIOR_TINT["sh"])}
    V10 = rf.COLOR["pcgs-v10"]
    mm = 1 / 25.4
    meta = json.load(open(f"{pdir}/meta.json"))
    A = json.load(open(metrics))
    P = 400

    def tight(imgs, top, bottom):
        a = np.zeros(imgs[0].shape[:2], bool)
        for im in imgs:
            a |= im[..., 3] > 0.02
        ys, xs = np.nonzero(a)
        m = int(0.02 * a.shape[0])
        x0, x1 = xs.min() - m, xs.max() + m
        w = x1 - x0
        return slice(ys.min() - int(top * w), ys.max() + int(bottom * w)), slice(x0, x1)

    maps = []
    for body, prior in (("itokawa", "tv"), ("ryugu", "shadow")):
        imgs = {t: np.pad(np.load(f"{pdir}/{body}_{t}_err.npy").astype(np.float32), ((P, P), (P, P), (0, 0)))
                for t in ("pcgs-v10", "pcgs-v18")}
        sy, sx = tight(list(imgs.values()), 0.07 if body == "ryugu" else 0.03, 0.13 if body == "ryugu" else 0.07)
        maps.append((body, prior, {t: np.clip(v[sy, sx][..., :3] * v[sy, sx][..., 3:], 0, 1) for t, v in imgs.items()},
                     (sx.stop - sx.start) / (sy.stop - sy.start)))
    rows = []
    for body, name, ph in CAMPAIGNS:
        for s in sorted(set(A.get(body, {}).get("pcgs-v10", {})) & set(A.get(body, {}).get("pcgs-v18", {}))):
            a, b = A[body]["pcgs-v10"][s], A[body]["pcgs-v18"][s]
            if "sfm_t20" in a and "sfm_t20" in b:
                rows.append((f"{name} {ph}°" + (f", split {s}" if body in ("itokawa", "ryugu") else ""), a, b,
                             "tv" if ph < 15 else "shadow"))

    W = rf.WMM * mm
    head, g = 3.4 * mm, 0.45 * mm
    tw = 23.6 * mm                                     # campaign tiles
    band = 5.2 * mm
    rh = 3.25 * mm
    hh = rh * len(rows)
    top_h = max(head + tw + band, head + 2.2 * mm + hh + 10.0 * mm)
    cb_w = 9.0 * mm
    hm = (W - cb_w - 3 * g - 1.6 * mm) / (2 * sum(m[3] for m in maps))
    H = top_h + 4.0 * mm + head + hm
    fig = plt.figure(figsize=(W, H), facecolor="white")

    def ax_at(x, y_top, w, h, off=True):
        ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
        if off:
            ax.set_axis_off()
        return ax

    # a: one held-out image per campaign, the prior used at that phase underneath
    y = H - head
    fig.text(0, (H - head / 2) / H, "a", fontsize=8, fontweight="bold", va="center", color=INK)
    fig.text(3.4 * mm / W, (H - head / 2) / H, "Campaigns by phase angle", fontsize=6.5, va="center", color=INK)
    for k, (body, name, ph) in enumerate(CAMPAIGNS):
        ax = ax_at(k * (tw + g), y, tw, tw)
        ax.imshow(np.load(f"{pdir}/{body}_thumb.npy"), cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        rf.tile_text(ax, 0.05, 0.96, f"{name} {ph}°", size=6.2, box=True)
    for k0, k1, key in ((0, 2, "tv"), (2, 4, "shadow")):
        lab, col, _ = PRIOR[key]
        x0, x1 = k0 * (tw + g), k1 * (tw + g) - g
        yb = y - tw - 1.3 * mm
        fig.add_artist(Line2D([x0 / W, x1 / W], [yb / H] * 2, color=col, lw=2.2, solid_capstyle="butt"))
        fig.text((x0 + x1) / 2 / W, (yb - 0.9 * mm) / H, f"{lab} (phase {'<' if key == 'tv' else '≥'} 15°)",
                 ha="center", va="top", fontsize=6.2, color=INK)

    # b: V10 -> this work, every campaign and split
    keys = [("rmse", "RMSE (m)", False), ("f1", "F-score at 1 m", True), ("normal", "Normal error (°)", False)]
    xb = 4 * (tw + g) + 4.6 * mm
    lab_w = 19.5 * mm
    pw = (W - xb - lab_w - 2 * 2.4 * mm) / 3
    fig.text(xb / W, (H - head / 2) / H, "b", fontsize=8, fontweight="bold", va="center", color=INK)
    fig.text((xb + 3.4 * mm) / W, (H - head / 2) / H, "Development runs, no night-side band: no prior → rule-selected prior",
             fontsize=6.5, va="center", color=INK)
    yt = y - 4.4 * mm
    for j, (k, lab, higher) in enumerate(keys):
        ax = ax_at(xb + lab_w + j * (pw + 2.4 * mm), yt, pw, hh, off=False)
        for yy, (nm, a, b, pr) in enumerate(rows):
            ax.axhspan(yy - 0.5, yy + 0.5, color=PRIOR[pr][2], lw=0, zorder=0)
            va, vb = rf.mval(a, k), rf.mval(b, k)
            better = (vb > va) if higher else (vb < va)
            ax.plot([va, vb], [yy, yy], color=rf.COLOR["pcgs-v18n"] if better else "#9b9b9b", lw=1.0, zorder=2,
                    solid_capstyle="round")
            ax.scatter([va], [yy], s=9, facecolor="white", edgecolor=V10, lw=0.8, zorder=3)
            ax.scatter([vb], [yy], s=11, color=rf.COLOR["pcgs-v18n"], edgecolor="white", lw=0.3, zorder=4)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows] if j == 0 else [], fontsize=5.5)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.tick_params(axis="x", length=2, width=0.5, labelsize=5.4, pad=1)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.set_xlabel(f"{lab} {'↑' if higher else '↓'}", fontsize=6.0, labelpad=1.2, color=INK)
    hl = [Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=V10, mew=0.8, label="no prior"),
          Line2D([], [], ls="", marker="o", ms=3.3, mfc=rf.COLOR["pcgs-v18n"], mec="white", mew=0.3, label="with prior"),
          Line2D([], [], color=rf.COLOR["pcgs-v18n"], lw=1.0, label="better"),
          Line2D([], [], color="#9b9b9b", lw=1.0, label="worse")]
    fig.legend(handles=hl, loc="upper center", bbox_to_anchor=((xb + lab_w + (W - xb - lab_w) / 2) / W,
                                                                (yt - hh - 6.6 * mm) / H),
               ncol=4, frameon=False, fontsize=5.6, handletextpad=0.3, columnspacing=1.3, borderaxespad=0)

    # c: error maps V10 vs this work, on the cameras of Fig. 2
    y = H - top_h - 4.0 * mm
    x = 0.0
    letter = iter("cd")
    for body, prior, imgs, asp in maps:
        wm = hm * asp
        name = "Itokawa" if body == "itokawa" else "Ryugu"
        lab, col, _ = PRIOR[prior]
        fig.text((x + wm + g / 2) / W, (y - head / 2) / H, f"{name}, {lab[0].lower() + lab[1:]}", ha="center", va="center",
                 fontsize=6.5, color=INK, fontweight="bold")
        for i, (t, tl) in enumerate((("pcgs-v10", "No prior"), ("pcgs-v18", "With prior"))):
            ax = ax_at(x, y - head, wm, hm)
            ax.imshow(imgs[t], interpolation="lanczos", aspect="auto")
            st = meta[body]["stats"][t]
            if i == 0:
                rf.tile_text(ax, 0.03 / asp, 0.975, next(letter), size=8, bold=True)
            rf.tile_text(ax, 0.97, 0.975, tl, size=6.2, ha="right", bold=(t == "pcgs-v18"), box=True)
            rf.tile_text(ax, 0.03 / asp, 0.025, f"mean |d| {st['mean_abs']:.2f} m\nRMSE {st['rmse']:.2f} m", size=5.5,
                         va="bottom", bold=(t == "pcgs-v18"), linespacing=1.2, box=True)
            x += wm + g
        x += 1.6 * mm - g
    cax = fig.add_axes([(W - cb_w + 1.4 * mm) / W, (y - head - 0.9 * hm) / H, 1.3 * mm / W, 0.8 * hm / H])
    cb = fig.colorbar(ScalarMappable(Normalize(-VMAX, VMAX), "RdBu_r"), cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([-3, 0, 3])
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
    cb.set_label("Signed distance d (m)", fontsize=6, labelpad=1)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
    print("wrote", out, f"{H / mm:.0f} mm high")


if __name__ == "__main__":
    {"render": lambda: render(sys.argv[2]),
     "compose": lambda: compose(sys.argv[2], sys.argv[3], sys.argv[4])}[sys.argv[1]]()
