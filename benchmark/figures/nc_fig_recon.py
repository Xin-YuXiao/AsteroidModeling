"""Fig. 2 (Nature Communications layout, 180 mm): reconstructions of Itokawa and Ryugu.

Per body: a held-out spacecraft image (test view of split 0), then the reference, PCGS and the
two strongest baselines rendered from that image's camera with the same clay material and
lighting (render_hq.py); below, the distributions of the signed distance to the reference and
the signed distance mapped on the same view (shared colour scale). A body that is tall in the
image is shown rotated by 90 deg (image and renders alike). Meshes: split 0, shared SfM alignment +
Taubin-20, frame registration (paper_figs.load_rec). Panels are rendered once and then composed.

usage: python nc_fig_recon.py render <panel dir>                          (CPU)
       python nc_fig_recon.py compose <panel dir> <out> [profiles dir]    (matplotlib only)
"""
import json
import os
import sys

import numpy as np

SPLIT = 0
VMAX = 3.0
BODIES = {
    "itokawa": dict(title="Itokawa", inst="Hayabusa AMICA", bar=100, zbar=20, test="ST_2421076883_v.png"),
    "ryugu": dict(title="Ryugu", inst="Hayabusa2 ONC-T", bar=200, zbar=50, test=None),
    # Bennu: panels from aa_render.py (largest full-body held-out image; Asteroid-GS with its scene extent)
    "bennu": dict(title="Bennu", inst="OSIRIS-REx MapCam", bar=100, zbar=20, test=None),
}
METHODS = [("ref", "Reference", None), ("pcgs", "PCGS (this work)", "pcgs-v18n_s{s}/mesh_mvpoisson_fairoff"),
           ("pgsr", "PGSR", "pgsr_s{s}/mesh_pgsr"), ("ags", "Asteroid-GS", "ags_s{s}/mesh_ags")]
LABEL = {k: l for k, l, _ in METHODS}
PROFILE_TAG = {"pcgs": "pcgs-v18n", "pgsr": "pgsr", "ags": "ags"}
PROFILE_BODY = {("bennu", "ags"): "ags-camext"}   # Asteroid-GS ran on Bennu only with its scene extent
COL = {"pcgs": "#d95926", "pgsr": "#2a78d6", "ags": "#1baf7a"}
# low raking key light and little ambient: relief (boulders, ridges) reads at figure size
LIGHT = dict(key=(-0.88, 0.40, 0.25), key_w=1.05, fill_w=0.06, amb_w=0.10, ao_rays=24)
# error maps: the earlier, less raking light, so that shading does not compete with the colour scale
ERR_LIGHT = dict(key=(-0.75, 0.55, 0.38), key_w=0.72, fill_w=0.16, amb_w=0.42)
ZOOM_PX = 150           # zoom window width in pixels of the held-out image (so the image crop stays legible)
ZOOM_ASPECT = 1.6        # width / height of the zoom tiles


def camera_view(pf, raycast, body, name=None, split=SPLIT):
    """Orthographic view matching a held-out camera: (view dir, up, half width, rotated?, camera dict)."""
    root = pf.D + f"/data/img/{body}"
    names = [l.strip() for l in open(f"{root}/scene_split{split}/test_images.txt") if l.strip()]
    name = name or sorted(names)[0]
    name = name if name.endswith(".png") else name + ".png"
    cam = next(c for c in raycast.load_cameras(f"{root}/scene_split{split}", "test") if c["name"] == name)
    ys, xs = np.nonzero(cam["mask"])
    rot = bool(ys.ptp() > 1.2 * xs.ptp())             # tall body: rotate image and renders by 90 deg
    T = pf.frame_tf(body)[:3, :3]
    Rb = cam["R"] @ np.linalg.inv(T / np.cbrt(np.linalg.det(T)))   # camera axes in the reference frame
    d, up = -Rb[2], (Rb[0] if rot else -Rb[1])
    ref = pf.load_ref(body)
    half = 1.03 * float(np.linalg.norm(ref.vertices, axis=1).max())
    return d, up, half, rot, cam, ref


def render(pdir):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paper_figs as pf
    import raycast
    import render_hq as hq
    pf.SUFFIX, pf.THREADS = "_sfm_t20", 12
    hq.CLAY[:] = (0.60, 0.585, 0.56)
    os.makedirs(pdir, exist_ok=True)
    image_only = bool(os.environ.get("NC_IMAGE_ONLY"))         # keep the renders, redo the image entries
    meta = json.load(open(f"{pdir}/meta.json")) if image_only else {}
    for body, cfg in BODIES.items():
        ref = pf.load_ref(body)
        root = pf.D + f"/data/img/{body}"
        names = [l.strip() for l in open(f"{root}/scene_split{SPLIT}/test_images.txt") if l.strip()]
        name = cfg["test"] or sorted(names)[0]
        name = name if name.endswith(".png") else name + ".png"
        cam = next(c for c in raycast.load_cameras(f"{root}/scene_split{SPLIT}", "test") if c["name"] == name)
        ys, xs = np.nonzero(cam["mask"])
        rot = bool(ys.ptp() > 1.2 * xs.ptp())             # tall body: rotate image and renders by 90 deg
        T = pf.frame_tf(body)[:3, :3]
        Rb = cam["R"] @ np.linalg.inv(T / np.cbrt(np.linalg.det(T)))   # camera axes in the reference frame
        d, up = -Rb[2], (Rb[0] if rot else -Rb[1])
        half = 1.03 * float(np.linalg.norm(ref.vertices, axis=1).max())
        Tf = pf.frame_tf(body)
        Xc = cam["R"] @ (np.linalg.solve(Tf, np.array([0.0, 0, 0, 1]))[:3] - cam["C"])
        px_m = float(np.cbrt(np.linalg.det(Tf[:3, :3])) * Xc[2] * 1000 / cam["fx"])
        if not image_only:
            meta[body] = dict(half=half, res=1500, rot=rot, stats={})
        for key, _, spec in ([] if image_only else METHODS):
            m = ref if spec is None else pf.load_rec(body, spec.format(s=SPLIT))
            r = hq.Renderer(m.vertices, m.faces, 12)
            clay = r.render(d, up, half, 1500, **LIGHT)
            np.save(f"{pdir}/{body}_{key}_clay.npy", clay.astype(np.float16))
            if key == "ref":
                meta[body]["zoom"] = zoom_window(clay, half, ZOOM_PX * px_m)
            zx, zy, zw = meta[body]["zoom"]
            np.save(f"{pdir}/{body}_{key}_zoom.npy",
                    r.render(d, up, half, 1000, window=(zx, zy, zw), **LIGHT).astype(np.float16))
            if spec is not None:
                sd = pf.signed_dist(ref, m.vertices)
                meta[body]["stats"][key] = dict(mean_abs=float(np.abs(sd).mean()), rmse=float(np.sqrt((sd ** 2).mean())),
                                                over3=float((np.abs(sd) > VMAX).mean()))
                img = r.render(d, up, half, 1500, colors=hq.diverging(sd, VMAX),
                               **ERR_LIGHT)
                np.save(f"{pdir}/{body}_{key}_err.npy", img.astype(np.float16))
            print(body, key, flush=True)
            del r, m
        # where the body centre (reference origin) falls in the image, and the image scale there,
        # so that the composition can resample the image onto the render grid
        np.save(f"{pdir}/{body}_image.npy", (cam["img"] * 255).astype(np.uint8))
        meta[body]["image"] = dict(name=name, utc=cam["utc"], phase=cam["phase"],
                                   u0=float(cam["fx"] * Xc[0] / Xc[2] + cam["cx"] - 0.5),
                                   v0=float(cam["fy"] * Xc[1] / Xc[2] + cam["cy"] - 0.5), px_m=px_m)
    json.dump(meta, open(f"{pdir}/meta.json", "w"), indent=1)


def zoom_window(clay, half, w):
    """Zoom window (centre x, centre y, width; view-plane metres) of width w on the reference render:
    the ZOOM_ASPECT box with the most fine relief (high-pass energy of the shading), away from the limb.
    The criterion looks at the reference only, never at a reconstruction."""
    import cv2
    res = clay.shape[0]
    s = 2 * half / res
    lum = clay[..., :3].mean(-1).astype(np.float32)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 0.006 * res)
    bw, bh = int(round(w / s)), int(round(w / s / ZOOM_ASPECT))
    inside = cv2.erode((clay[..., 3] > 0.5).astype(np.uint8), np.ones((bh, bw), np.uint8))
    e = cv2.boxFilter(hp ** 2 * (clay[..., 3] > 0.5), -1, (bw, bh), normalize=False)
    e[inside == 0] = -1                                           # box centre must keep the whole box on the body
    iy, ix = np.unravel_index(np.argmax(e), e.shape)
    return [float(-half + (ix + 0.5) * s), float(half - (iy + 0.5) * s), float(w)]


# ---------------------------------------------------------------- composition
def img_affine(ii, rot, x0, y0, step):
    """Inverse map for cv2.warpAffine: output pixel (X, Y) of a view-plane grid with top-left corner
    (x0, y0) and pixel size `step` (metres) -> pixel of the spacecraft image."""
    px = ii["px_m"]
    if rot:   # render right = image down, render up = image right
        return np.float32([[0, -step / px, ii["u0"] + (y0 - 0.5 * step) / px],
                           [step / px, 0, ii["v0"] + (x0 + 0.5 * step) / px]])
    return np.float32([[step / px, 0, ii["u0"] + (x0 + 0.5 * step) / px],
                       [0, step / px, ii["v0"] - (y0 - 0.5 * step) / px]])


def compose(pdir, out, profiles=None):
    """Per body three rows of tiles on black: the held-out image and the four shapes from its camera,
    the zoom window of the same five, then the signed-distance density and the error maps."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_figs as rf
    rf.style()
    meta = json.load(open(f"{pdir}/meta.json"))
    mm = 1 / 25.4
    P = 400                                                              # zero border around the renders
    ZC = rf.ZOOM

    def on_black(rgba):
        return np.clip(rgba[..., :3] * rgba[..., 3:], 0, 1)

    def tight(imgs, top, bottom):
        """Crop to the union of the silhouettes, with black strips above and below (fractions of the
        width) that carry the text, so that no label sits on the body."""
        a = np.zeros(imgs[0].shape[:2], bool)
        for im in imgs:
            a |= im[..., 3] > 0.02
        ys, xs = np.nonzero(a)
        m = int(0.02 * a.shape[0])
        x0, x1 = xs.min() - m, xs.max() + m
        w = x1 - x0
        return slice(ys.min() - int(top * w), ys.max() + int(bottom * w)), slice(x0, x1)

    def image_stretch(img):
        hi = np.percentile(img[img > 0.03], 99.6)
        return np.repeat(np.clip(img / hi, 0, 1)[..., None], 3, -1)

    blocks = []
    for body in BODIES:
        mt, ii = meta[body], meta[body]["image"]
        pad = lambda a: np.pad(a, ((P, P), (P, P), (0, 0)))           # noqa: E731
        clay = {k: pad(np.load(f"{pdir}/{body}_{k}_clay.npy").astype(np.float32)) for k, *_ in METHODS}
        err = {k: pad(np.load(f"{pdir}/{body}_{k}_err.npy").astype(np.float32)) for k, _, sp in METHODS if sp}
        sy, sx = tight(list(clay.values()) + list(err.values()), 0.035, 0.085)
        sr = 2 * mt["half"] / mt["res"]
        x0, y0 = -mt["half"] + (sx.start - P) * sr, mt["half"] - (sy.start - P) * sr
        im = np.load(f"{pdir}/{body}_image.npy").astype(np.float32) / 255
        full_img = cv2.warpAffine(im, img_affine(ii, mt["rot"], x0, y0, sr), (sx.stop - sx.start, sy.stop - sy.start),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        # zoom band: central 1/ZOOM_ASPECT of the square zoom renders
        zx, zy, zw = mt["zoom"]
        zoom = {k: np.load(f"{pdir}/{body}_{k}_zoom.npy").astype(np.float32) for k, *_ in METHODS}
        n = next(iter(zoom.values())).shape[0]
        nb = int(round(n / ZOOM_ASPECT))
        yo = (n - nb) // 2
        zoom = {k: on_black(v[yo:yo + nb]) for k, v in zoom.items()}
        step = zw / n
        zimg = cv2.warpAffine(im, img_affine(ii, mt["rot"], zx - zw / 2, zy + zw / 2 - yo * step, step), (n, nb),
                              flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        hi = np.percentile(full_img[full_img > 0.03], 99.6)
        box = ((zx - zw / 2 - x0) / sr, (y0 - (zy + zw / ZOOM_ASPECT / 2)) / sr, zw / sr, zw / ZOOM_ASPECT / sr)
        blocks.append(dict(body=body, image=image_stretch(full_img), zimage=np.repeat(np.clip(zimg / hi, 0, 1)[..., None], 3, -1),
                           clay={k: on_black(v[sy, sx]) for k, v in clay.items()},
                           err={k: on_black(v[sy, sx]) for k, v in err.items()}, zoom=zoom, box=box,
                           asp=(sx.stop - sx.start) / (sy.stop - sy.start), px=sr, zstep=step))

    W = rf.WMM * mm
    lab_w, cb_w, g, gb, head = 4.6 * mm, 9.0 * mm, 0.45 * mm, 2.6 * mm, 3.4 * mm
    pw = (W - lab_w - cb_w - 4 * g) / 5
    hz = pw / ZOOM_ASPECT
    rows_h = [(pw / b["asp"], hz, pw / b["asp"]) for b in blocks]
    H = head + sum(sum(r) + 2 * g for r in rows_h) + gb * (len(blocks) - 1)
    fig = plt.figure(figsize=(W, H), facecolor="white")

    def ax_at(x, y_top, w, h):
        ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
        ax.set_axis_off()
        return ax

    def scale_bar(ax, L, label, frac_y, plate=False):
        hh, ww = ax.images[0].get_array().shape[:2]
        x1, yb = 0.05 * ww, frac_y * hh
        if plate:                              # on bright tiles: a dark plate behind bar and label
            ax.add_patch(Rectangle((x1 - 0.025 * ww, yb - 0.075 * hh), L + 0.17 * ww, 0.15 * hh, fc="black",
                                   ec="none", alpha=0.6, zorder=3))
        ax.plot([x1, x1 + L], [yb] * 2, color="white", lw=1.0, solid_capstyle="butt", zorder=4)
        ax.text(x1 + L + 0.02 * ww, yb, label, color="white", fontsize=5.5, ha="left", va="center", zorder=4)

    letters = iter("abcdefghijkl")
    cols = ["Held-out image", "Reference", "PCGS (this work)", "PGSR", "Asteroid-GS"]
    for j, t in enumerate(cols):
        fig.text((lab_w + j * (pw + g) + pw / 2) / W, (H - head / 2) / H, t, ha="center", va="center", fontsize=6.5,
                 color=rf.INK, fontweight="bold" if j == 2 else "normal")
    y = H - head
    for b, (h1, h2, h3) in zip(blocks, rows_h):
        cfg, mt = BODIES[b["body"]], meta[b["body"]]
        fig.text(lab_w * 0.42 / W, (y - (h1 + h2 + h3 + 2 * g) / 2) / H, cfg["title"], rotation=90, va="center",
                 ha="center", fontsize=7, fontweight="bold", color=rf.INK)
        keys = ["image", "ref", "pcgs", "pgsr", "ags"]
        # row 1: full views, zoom window marked
        for j, k in enumerate(keys):
            ax = ax_at(lab_w + j * (pw + g), y, pw, h1)
            ax.imshow(b["image"] if k == "image" else b["clay"][k], interpolation="lanczos", aspect="auto")
            ax.add_patch(Rectangle(b["box"][:2], *b["box"][2:], fill=False, ec=ZC, lw=0.6))
            if j == 0:
                rf.tile_text(ax, 0.03, 0.975, next(letters), size=8, bold=True)
                rf.tile_text(ax, 0.5, 0.018, f"{cfg['inst']}, phase {mt['image']['phase']:.0f}°", size=5.5,
                             va="bottom", ha="center")
            if k == "ref":
                scale_bar(ax, cfg["bar"] / b["px"], f"{cfg['bar']} m", 0.955)
        y -= h1 + g
        # row 2: the zoom window, framed in the zoom colour
        for j, k in enumerate(keys):
            ax = ax_at(lab_w + j * (pw + g), y, pw, h2)
            ax.imshow(b["zimage"] if k == "image" else b["zoom"][k], interpolation="lanczos", aspect="auto")
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=ZC, lw=1.0))
            if j == 0:
                rf.tile_text(ax, 0.03, 0.955, next(letters), size=8, bold=True, box=True)
            if k == "ref":
                zb = cfg["zbar"]
                scale_bar(ax, zb / b["zstep"], f"{zb} m", 0.9, plate=True)
        y -= h2 + g
        # row 3: signed-distance density over the first two tiles (a plot on white), then the error maps
        fig.text((lab_w + 0.6 * mm) / W, (y - 3.0 * mm) / H, next(letters), fontsize=8, fontweight="bold",
                 va="baseline", ha="left", color=rf.INK)
        axd = fig.add_axes([(lab_w + 12.0 * mm) / W, (y - h3 + 6.4 * mm) / H, (2 * pw + g - 14.0 * mm) / W,
                            (h3 - 8.4 * mm) / H])
        for sp in ("top", "right"):
            axd.spines[sp].set_visible(False)
        axd.tick_params(length=2, width=0.5, pad=1.2, labelsize=5.5, colors=rf.INK)
        top = 0
        for k in ("ags", "pgsr", "pcgs"):
            f = os.path.join(profiles or "", f"{b['body']}__{PROFILE_BODY.get((b['body'], k), PROFILE_TAG[k])}__s{SPLIT}.json")
            if not os.path.exists(f):
                continue
            hst = json.load(open(f))["signed"]
            c = np.convolve(np.asarray(hst["counts"], float), np.ones(3) / 3, mode="same")
            x = hst["lo"] + hst["step"] * (np.arange(len(c)) + 0.5)
            sel = np.abs(x) <= VMAX
            yv = c[sel] / (c.sum() + hst["under"] + hst["over"]) / hst["step"]
            axd.plot(x[sel], yv, color=rf.COLOR[PROFILE_TAG[k]], lw=1.2 if k == "pcgs" else 0.85,
                     zorder=3 if k == "pcgs" else 2)
            top = max(top, yv.max())
        axd.axvline(0, color=rf.INK2, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
        axd.set_xlim(-VMAX, VMAX)
        axd.set_ylim(0, top * 1.08)
        axd.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        axd.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.3 if h3 < 22 * mm else 0.2))
        axd.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
        axd.set_xlabel("Signed distance d (m)", fontsize=6, labelpad=1, color=rf.INK)
        axd.set_ylabel("Density (1/m)", fontsize=6, labelpad=1.5, color=rf.INK)
        hl = [Line2D([], [], color=rf.COLOR[PROFILE_TAG[k]], lw=1.4 if k == "pcgs" else 1.0, label=t)
              for k, t in (("pcgs", "PCGS"), ("pgsr", "PGSR"), ("ags", "Asteroid-GS"))]
        axd.legend(handles=hl, loc="upper right", frameon=False, fontsize=5.6, handlelength=1.3, handletextpad=0.4,
                   borderaxespad=0.0, labelspacing=0.25)
        for j, k in enumerate(("pcgs", "pgsr", "ags")):
            ax = ax_at(lab_w + (j + 2) * (pw + g), y, pw, h3)
            ax.imshow(b["err"][k], interpolation="lanczos", aspect="auto")
            st = mt["stats"][k]
            if j == 0:
                rf.tile_text(ax, 0.03, 0.975, next(letters), size=8, bold=True)
            rf.tile_text(ax, 0.5, 0.018, f"mean |d| {st['mean_abs']:.2f} m   RMSE {st['rmse']:.2f} m", size=5.5,
                         color="white", va="bottom", ha="center", bold=(k == "pcgs"))
        cax = fig.add_axes([(W - cb_w + 1.4 * mm) / W, (y - 0.9 * h3) / H, 1.3 * mm / W, 0.8 * h3 / H])
        cb = fig.colorbar(ScalarMappable(Normalize(-VMAX, VMAX), "RdBu_r"), cax=cax, orientation="vertical")
        cb.outline.set_linewidth(0.4)
        cb.set_ticks([-3, 0, 3] if h3 < 22 * mm else [-3, -2, -1, 0, 1, 2, 3])
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
        cb.set_label("d (m)", fontsize=6, labelpad=1)
        y -= h3 + gb
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
    print("wrote", out, f"{W / mm:.0f} x {H / mm:.0f} mm")


if __name__ == "__main__":
    {"render": lambda: render(sys.argv[2]),
     "compose": lambda: compose(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)}[sys.argv[1]]()
