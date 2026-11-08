"""Fig. 1 (Nature Communications layout, 183 mm): photoclinometric Gaussian splatting.

a  observation geometry (incidence i, emission e, phase angle alpha)
b  pipeline as stage cards with real tiles: spacecraft images + SPICE Sun -> 2D Gaussian surfels ->
   photometric rendering (albedo x lunar-Lambert shading x Sun visibility) -> mesh; the objective and
   the phase-dependent prior below (colours shared with Fig. 5)
c  decomposition of a held-out Ryugu image by the trained model (pcgs_decompose.py): observed,
   predicted, albedo, shading, surface normals, residual
usage: python nc_fig_overview.py <decomposition dir> <panel dir of nc_fig_recon.py> <out>
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402,F401
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon, Arc, Ellipse  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import result_figs as rf  # noqa: E402

INK, INK2 = rf.INK, rf.INK2
ACC = rf.COLOR["pcgs-v18n"]
SUN, VIEW_C = "#c9962a", "#3d6fa8"
TV, SHADOW = (rf.PRIOR["tv"], rf.PRIOR_TINT["tv"]), (rf.PRIOR["sh"], rf.PRIOR_TINT["sh"])   # as Figs. 3, 5
CARD = "#f3f1ed"
BODY, VIEW = "ryugu", "hyb2_onc_20180710_060508_tvf_l2c"
SURF_W = 120.0                   # surfel window (m)
mm = 1 / 25.4


def style():
    rf.style()
    plt.rcParams.update({"font.size": 6.3, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
                         "mathtext.cal": "Liberation Sans:italic"})


def crop_box(alpha, margin=0.06):
    ys, xs = np.nonzero(alpha > 0.5)
    c, s = ((ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2), max(np.ptp(ys), np.ptp(xs)) * (0.5 + margin)
    return slice(int(c[0] - s), int(c[0] + s)), slice(int(c[1] - s), int(c[1] + s))


def geometry(ax):
    """Sun, surface element with normal, camera; angles i, e, alpha."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 8.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    P = np.array([5.0, 1.9])                                  # surface point
    xs = np.linspace(0.9, 9.1, 200)
    ys = 1.9 + 0.10 * np.sin(3.1 * xs) + 0.05 * np.sin(7.3 * xs) - 0.012 * (xs - 5) ** 2
    ax.fill_between(xs, 0.7, ys, color="#dcd6cc", lw=0)
    ax.plot(xs, ys, color="#8c8780", lw=0.7)
    n = np.array([0.0, 1.0])
    sun = np.array([-np.sin(np.radians(42)), np.cos(np.radians(42))])
    cam = np.array([np.sin(np.radians(24)), np.cos(np.radians(24))])
    L = 4.2

    def arrow(a, b, c, lw=1.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=c, lw=lw, shrinkA=0, shrinkB=0))

    arrow(P, P + 3.1 * n, INK2, 0.7)
    ax.text(*(P + 3.25 * n), "n", fontsize=6.5, style="italic", color=INK2, ha="center", va="bottom")
    arrow(P + L * sun, P, SUN, 1.1)
    arrow(P, P + L * cam, VIEW_C, 1.1)
    sc = P + (L + 0.6) * sun
    ax.add_patch(Circle(sc, 0.42, color="#f2c94c", lw=0, zorder=3))
    for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        d = np.array([np.cos(t), np.sin(t)])
        ax.plot(*np.stack([sc + 0.56 * d, sc + 0.78 * d]).T, color="#f2c94c", lw=0.7)
    ax.text(sc[0], sc[1] - 0.95, "Sun", fontsize=5.8, ha="center", va="top", color=INK2)
    cc = P + (L + 0.6) * cam
    rot = np.array([[np.cos(np.radians(-24)), -np.sin(np.radians(-24))],
                    [np.sin(np.radians(-24)), np.cos(np.radians(-24))]])
    ax.add_patch(Polygon(cc + np.array([[-0.38, -0.25], [0.38, -0.25], [0.38, 0.25], [-0.38, 0.25]]) @ rot.T,
                         closed=True, color="#4a5563", lw=0, zorder=3))
    ax.add_patch(Polygon(cc + np.array([[-0.18, -0.25], [0.18, -0.25], [0.28, -0.52], [-0.28, -0.52]]) @ rot.T,
                         closed=True, color="#4a5563", lw=0, zorder=3))
    ax.text(cc[0] + 0.62, cc[1] - 0.55, "camera", fontsize=5.8, ha="left", va="center", color=INK2)

    def ang(v):
        return np.degrees(np.arctan2(v[1], v[0]))

    for a0, a1, r, lab, c in ((ang(n), ang(sun), 1.1, "i", SUN), (ang(cam), ang(n), 1.4, "e", VIEW_C),
                              (ang(cam), ang(sun), 2.1, "α", INK2)):
        ax.add_patch(Arc(P, 2 * r, 2 * r, theta1=a0, theta2=a1, color=c, lw=0.6))
        m = np.radians((a0 + a1) / 2)
        ax.text(P[0] + (r + 0.25) * np.cos(m), P[1] + (r + 0.25) * np.sin(m), lab, fontsize=6.5, style="italic",
                color=c, ha="center", va="center")
    ax.text(5.0, 0.35, "$\\mu_0 = \\cos i,\\ \\ \\mu = \\cos e$", fontsize=6, ha="center", va="top", color=INK)


def surfels(ax):
    """2D Gaussian surfels on black: flat splats tangent to a surface, each with an albedo and a normal."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    xs = np.linspace(0.6, 9.4, 300)
    f = lambda x: 3.4 + 2.2 * np.exp(-((x - 4.6) / 2.6) ** 2) + 0.35 * np.sin(1.7 * x)   # noqa: E731
    ax.plot(xs, f(xs), color="#6b6b6b", lw=0.6, ls=(0, (2, 2)))
    rng = np.random.default_rng(3)
    for x in np.linspace(1.2, 8.8, 9):
        y = f(x)
        dy = (f(x + 1e-3) - f(x - 1e-3)) / 2e-3
        g = 0.45 + 0.45 * rng.random()
        ax.add_patch(Ellipse((x, y), 1.9, 0.42, angle=np.degrees(np.arctan(dy)), fc=(g, g * 0.97, g * 0.93),
                             ec="none", alpha=0.95))
        nn = np.array([-dy, 1.0]) / np.hypot(dy, 1.0)
        ax.add_patch(FancyArrowPatch((x, y), (x + 1.3 * nn[0], y + 1.3 * nn[1]), arrowstyle="-|>", mutation_scale=4,
                                     color=ACC, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(0.5, 0.06, "albedo $A$ (grey), normal $n$ (red)", transform=ax.transAxes, fontsize=5.3, ha="center",
            va="bottom", color="#d0d0d0")


def render(sdir):
    """Real-data panels for a and b (CPU): the Ryugu reference seen obliquely with the training
    camera directions (split 0) and the held-out camera of c; and the trained surfels of PCGS in a small
    window of the held-out view, as flat discs shaded by their albedo."""
    import paper_figs as pf
    import raycast
    import render_hq as hq
    import nc_fig_recon as rc
    from plyfile import PlyData
    pf.SUFFIX, pf.THREADS = "_sfm_t20", 12
    hq.CLAY[:] = (0.60, 0.585, 0.56)
    os.makedirs(sdir, exist_ok=True)
    Tf = pf.frame_tf(BODY)
    sc = float(np.cbrt(np.linalg.det(Tf[:3, :3])))
    Rt = Tf[:3, :3] / sc                                            # scene -> reference rotation
    ref = pf.load_ref(BODY)
    Rmax = float(np.linalg.norm(ref.vertices, axis=1).max())
    root = pf.D + f"/data/img/{BODY}/scene_split0"
    cams = raycast.load_cameras(root, "all")
    test = {l.strip() for l in open(f"{root}/test_images.txt") if l.strip()}
    out = dict(R=Rmax, rot_angle=float(np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))))
    C = np.array([(Tf @ np.r_[c["C"], 1])[:3] * 1000 for c in cams])               # metres, reference frame
    out["dirs"] = C / np.linalg.norm(C, axis=1, keepdims=True)
    out["axes"] = np.array([c["R"] @ Rt.T for c in cams])                          # rows: camera x, y, z
    out["phase"] = np.array([c["phase"] for c in cams])
    out["dist_km"] = np.linalg.norm(C, axis=1) / 1000
    out["is_test"] = np.array([c["name"] in test or c["name"][:-4] in test for c in cams])
    out["held"] = np.array([c["name"] == VIEW + ".png" for c in cams])
    hc = cams[int(np.argmax(out["held"]))]
    out["sun_held"] = Rt @ hc["sun"]
    # oblique view from 35 deg above the mean camera plane, lit by the Sun of the held-out image
    n = np.linalg.svd(out["dirs"] - out["dirs"].mean(0))[2][-1]
    n = n if n @ np.array([0, 0, 1.0]) > 0 else -n
    az = out["dirs"][out["held"]][0]
    az = az - (az @ n) * n
    az /= np.linalg.norm(az)
    side = np.cross(n, az)
    d = np.cos(np.radians(35)) * (np.cos(np.radians(40)) * az + np.sin(np.radians(40)) * side) + np.sin(np.radians(35)) * n
    up = n - (n @ d) * d
    up /= np.linalg.norm(up)
    fwd, right, up2 = hq._basis(d, up)
    s = out["sun_held"]
    key = (float(s @ right), float(s @ up2), float(-s @ fwd))
    half = 2.0 * Rmax
    r = hq.Renderer(ref.vertices, ref.faces, 12)
    img = r.render(d, up, half, 1400, key=key, key_w=1.05, fill_w=0.05, amb_w=0.10, ao_rays=16)
    out.update(scene=img.astype(np.float16), view=np.stack([right, up2, -fwd]), half=half)
    del r
    print("scene", out["rot_angle"], flush=True)
    # trained surfels in a window of the held-out view (same camera as Fig. 2)
    dh, uh, halfh, roth, cam, _ = rc.camera_view(pf, raycast, BODY, VIEW)
    ply = PlyData.read(f"{pf.D}/results/img/{BODY}/pcgs-v18n_s0/point_cloud/iteration_30000/point_cloud.ply")["vertex"]
    P = np.stack([ply["x"], ply["y"], ply["z"]], 1)
    keep = 1 / (1 + np.exp(-np.asarray(ply["opacity"]))) > 0.5
    P = (P @ Tf[:3, :3].T + Tf[:3, 3]) * 1000
    S = np.exp(np.stack([ply["scale_0"], ply["scale_1"]], 1)) * sc * 1000
    Q = np.stack([ply[f"rot_{k}"] for k in range(4)], 1)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    Rq = np.array([raycast._qrot(q) for q in Q])
    tu, tv = (Rq[:, :, 0] @ Rt.T), (Rq[:, :, 1] @ Rt.T)
    alb = np.clip(0.5 + 0.28209479 * np.asarray(ply["f_dc_0"]), 0, None)
    P, S, tu, tv, alb = P[keep], S[keep], tu[keep], tv[keep], alb[keep]
    # window: 90 m around the most textured part of the reference in the held-out view
    rr = hq.Renderer(ref.vertices, ref.faces, 12)
    zx, zy, _ = rc.zoom_window(rr.render(dh, uh, halfh, 600, ss=1, **rc.LIGHT), halfh, SURF_W)
    del rr
    fw, rw, uw = hq._basis(dh, uh)
    sel = (np.abs(P @ rw - zx) < SURF_W) & (np.abs(P @ uw - zy) < SURF_W) & (P @ -fw > 0)    # near side only
    k = 16
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, si = np.cos(ang), np.sin(ang)
    Ps, Ss, us, vs = P[sel], S[sel] * 1.0, tu[sel], tv[sel]             # 1-sigma footprint
    rim = Ps[:, None] + Ss[:, :1, None] * c[None, :, None] * us[:, None] + Ss[:, 1:, None] * si[None, :, None] * vs[:, None]
    V = np.concatenate([Ps[:, None], rim], 1).reshape(-1, 3)
    base = np.arange(len(Ps))[:, None] * (k + 1)
    F = np.stack([np.repeat(base, k, 1), base + 1 + np.arange(k), base + 1 + (np.arange(k) + 1) % k], -1).reshape(-1, 3)
    a = alb[sel] / np.percentile(alb[sel], 98) * 0.55
    rs = hq.Renderer(V, F, 12)
    col = np.repeat(np.stack([a, a * 0.975, a * 0.94], 1), k + 1, 0)
    out["surfels"] = rs.render(dh, uh, halfh, 900, colors=col, window=(zx, zy, SURF_W),
                               **{**rc.LIGHT, "amb_w": 0.22}).astype(np.float16)
    # the same discs coloured by their normal in the camera frame (mapping of the normal panel in c)
    nrm = np.cross(us, vs)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm[(nrm @ fw) > 0] *= -1
    rgb = np.stack([0.5 + 0.5 * (nrm @ rw), 0.5 + 0.5 * (nrm @ uw), 0.5 - 0.5 * (nrm @ fw)], 1)
    col = np.repeat(np.clip(rgb, 0, 1) ** 2.2, k + 1, 0)                     # sRGB -> linear
    out["surfels_normal"] = rs.render(dh, uh, halfh, 900, colors=col, window=(zx, zy, SURF_W), key=rc.LIGHT["key"],
                                      key_w=0.55, fill_w=0.1, amb_w=0.55, ao_rays=16).astype(np.float16)
    out["surfel_window"] = np.array([zx, zy, SURF_W])
    out["n_surfels"] = int(keep.sum())
    np.savez_compressed(f"{sdir}/scene.npz", **out)
    print("surfels", int(sel.sum()), "of", int(keep.sum()), flush=True)


def scene_panel(fig, ax, sc):
    """a: the reference lit by the Sun of the held-out image, every camera of split 0 as a frustum along
    its true direction (distance compressed), training views coloured by the prior their phase selects."""
    from matplotlib.patches import Rectangle
    img = sc["scene"].astype(np.float32)
    res, half, R = img.shape[0], float(sc["half"]), float(sc["R"])
    right, up2, toward = sc["view"]
    bw, bh = ax.get_position().width * fig.get_figwidth(), ax.get_position().height * fig.get_figheight()
    y_top = 0.13 * res                                              # the empty sky above the ring is cropped
    ylim = y_top + res * bh / bw
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="black", zorder=0))
    rgba = np.concatenate([np.clip(img[..., :3], 0, 1), img[..., 3:]], -1)
    ax.imshow(rgba, extent=(0, res, res, 0), zorder=2, interpolation="lanczos")
    ax.set_xlim(0, res)
    ax.set_ylim(ylim, y_top)

    def px(P):
        P = np.atleast_2d(P)
        return np.stack([(P @ right + half) / (2 * half) * res, (half - P @ up2) / (2 * half) * res], -1)

    rv, L, a = 1.5 * R, 0.17 * R, 0.065 * R
    order = np.argsort(sc["dirs"] @ toward)                         # far to near
    for i in order:
        Pc = sc["dirs"][i] * rv
        x, y, z = sc["axes"][i]
        base = Pc + L * z
        cr = np.array([base + a * (sx * x + sy * y) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        q, A = px(cr), px(Pc)[0]
        front = Pc @ toward > 0
        if sc["held"][i]:
            fc, ec, lw, zo = ACC, ACC, 0.7, 6
        elif sc["is_test"][i]:
            fc, ec, lw, zo = "none", "#e6e6e6", 0.45, 3 if front else 1
        else:
            fc = TV[0] if sc["phase"][i] < 15 else SHADOW[0]
            ec, lw, zo = "#1a1a1a", 0.2, 3 if front else 1
        ax.add_patch(Polygon(q, closed=True, fc=fc, ec=ec, lw=lw, zorder=zo, alpha=1.0 if front else 0.55))
        for c in q:
            ax.plot([A[0], c[0]], [A[1], c[1]], color=ec if fc == "none" or sc["held"][i] else "#cfcfcf",
                    lw=0.25, zorder=zo, alpha=1.0 if front else 0.55)
        if sc["held"][i]:
            ax.text(A[0], A[1] - 0.025 * res, "c", color="white", fontsize=6.5, fontweight="bold", ha="center",
                    va="bottom", zorder=7)
    # Sun of the held-out image: parallel rays towards the body
    s = sc["sun_held"]
    s2 = np.array([s @ right, -(s @ up2)])
    s2 /= max(np.linalg.norm(s2), 1e-6)
    ctr = px(np.zeros(3))[0]
    perp = np.array([-s2[1], s2[0]])
    Rp = R / (2 * half) * res
    for o in (-0.35, 0, 0.35):
        p0 = ctr + s2 * 1.62 * Rp + perp * o * Rp
        ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p0 - s2 * 0.42 * Rp), arrowstyle="-|>", mutation_scale=5,
                                     color=rf.ZOOM, lw=0.8, zorder=8))
    sp = ctr + s2 * 1.8 * Rp
    ax.add_patch(Circle(tuple(sp), 0.06 * Rp, color=rf.ZOOM, lw=0, zorder=8))
    ax.text(*(sp - s2 * 0.0 + np.array([0, -0.14 * Rp])), "Sun", color="white", fontsize=5.6, ha="center",
            va="bottom", zorder=8)
    # legend in the black strip below the scene: only the classes present
    train = ~sc["is_test"]
    items = ([(TV[0], "training view, phase < 15°")] if (sc["phase"][train] < 15).any() else []) +             ([(SHADOW[0], "training view, phase ≥ 15°")] if (sc["phase"][train] >= 15).any() else []) +             [("none", "test view"), (ACC, "image of c")]
    yl = 0.83 * res                                                    # below the camera ring
    for k, (c, t) in enumerate(items):
        xx, yy = 0.06 * res, yl + k * 0.068 * res
        ax.add_patch(Rectangle((xx, yy - 0.02 * res), 0.04 * res, 0.04 * res, fc=c,
                               ec="#e6e6e6" if c == "none" else c, lw=0.5, zorder=5))
        ax.text(xx + 0.06 * res, yy, t, color="white", fontsize=5.5, va="center", ha="left", zorder=5)
    ax.text(0.06 * res, yl + len(items) * 0.068 * res, f"{int(train.sum())} training, {int(sc['is_test'].sum())} test "
            f"views\n≈ {np.median(sc['dist_km']):.0f} km, not to scale", color="white", fontsize=5.2, ha="left",
            va="top", zorder=5, linespacing=1.3)


def main():
    if sys.argv[1] == "render":
        return render(sys.argv[2])
    ddir, pdir, sdir, out = sys.argv[1:5]
    sc = dict(np.load(os.path.join(sdir, "scene.npz"))) if os.path.exists(os.path.join(sdir, "scene.npz")) else None
    style()
    z = {k: v.astype(np.float32) for k, v in np.load(os.path.join(ddir, BODY, VIEW + ".npz")).items() if k != "test"}
    sy, sx = crop_box(z["alpha"])
    m = (z["alpha"] > 0.5)[sy, sx]
    meta = json.load(open(os.path.join(pdir, "meta.json")))[BODY]

    W = rf.WMM * mm
    g = 0.45 * mm
    # top: geometry card + four pipeline cards + objective bar
    aw, gap_ab, ga = 43 * mm, 4.4 * mm, 4.6 * mm
    xb = aw + gap_ab
    cw = (W - xb - 3 * ga) / 4
    pad, hdr = 1.6 * mm, 4.4 * mm
    tile = cw - 2 * pad
    cap = 6.6 * mm
    ch = hdr + tile + cap
    obj_h, obj_gap = 10.0 * mm, 2.2 * mm
    top_h = 4.0 * mm + ch + obj_gap + obj_h
    # bottom: decomposition tiles
    cb_w, head = 11.5 * mm, 3.6 * mm
    pw = (W - cb_w - 5 * g) / 6
    H = top_h + 5.0 * mm + head + pw + 0.3 * mm
    fig = plt.figure(figsize=(W, H), facecolor="white")

    def ax_at(x, y_top, w, h):
        ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
        ax.set_axis_off()
        return ax

    def card(x, y_top, w, h, fc=CARD, r=1.2 * mm):
        fig.add_artist(FancyBboxPatch((x / W, (y_top - h) / H), w / W, h / H, transform=fig.transFigure,
                                      boxstyle=f"round,pad=0,rounding_size={r / H}", fc=fc, ec="none",
                                      mutation_aspect=H / W, zorder=-10))

    def letter(x, y, s, title=None):
        fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
        if title:
            fig.text((x + 3.4 * mm) / W, (y - 0.2 * mm) / H, title, fontsize=6.5, va="top", ha="left", color=INK)

    top = H
    # ---- a: geometry
    if sc is None:
        letter(0, top, "a", "Observation geometry")
        card(0, top - 4.0 * mm, aw, ch + obj_gap + obj_h)
        geometry(ax_at(1.0 * mm, top - 4.6 * mm, aw - 2.0 * mm, ch + obj_gap + obj_h - 2.0 * mm))
    else:
        letter(0, top, "a", "Views and illumination")
        scene_panel(fig, ax_at(0, top - 4.0 * mm, aw, ch + obj_gap + obj_h), sc)

    # ---- b: pipeline cards
    letter(xb, top, "b", "Photometrically constrained Gaussian splatting (PCGS)")
    clay = np.pad(np.load(os.path.join(pdir, f"{BODY}_pcgs_clay.npy")).astype(np.float32), ((300, 300), (300, 300), (0, 0)))
    ys, xs = np.nonzero(clay[..., 3] > 0.02)
    cs = max(np.ptp(ys), np.ptp(xs)) * 0.56
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    cc = clay[int(cy - cs):int(cy + cs), int(cx - cs):int(cx + cs)]
    clay_c = np.clip(cc[..., :3] * (0.66 / 0.60) * cc[..., 3:], 0, 1)
    stages = [("1  Input", "spacecraft images $I$\n+ Sun direction, camera pose", "img"),
              ("2  Representation", "2D Gaussian surfels\nwith albedo and normal", "surf"),
              ("3  Photometric rendering", "Î = A · Φ(μ₀, μ; L) · V\nlunar-Lambert Φ, Sun visibility V", "pred"),
              ("4  Surface", "watertight mesh from\nmulti-view depth fusion", "mesh")]
    yc = top - 4.0 * mm
    for k, (head_t, txt, kind) in enumerate(stages):
        x = xb + k * (cw + ga)
        card(x, yc, cw, ch)
        fig.text((x + pad) / W, (yc - hdr / 2) / H, head_t, fontsize=6.3, fontweight="bold", va="center", ha="left",
                 color=INK)
        ax = ax_at(x + pad, yc - hdr, tile, tile)
        if kind == "img":
            ax.imshow(z["gt"][sy, sx], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        elif kind == "surf" and sc is not None:
            sf = sc["surfels_normal"] if "surfels_normal" in sc else sc["surfels"]
            ax.imshow(np.clip(sf[..., :3].astype(np.float32) * sf[..., 3:], 0, 1), interpolation="lanczos")
            rf.tile_text(ax, 0.5, 0.03, f"trained surfels, {sc['surfel_window'][2]:.0f} m", size=5.2,
                         ha="center", va="bottom", color="white", box=True)
        elif kind == "surf":
            ax.set_axis_on()
            surfels(ax)
        elif kind == "pred":
            ax.imshow(np.where(m, z["pred"][sy, sx], 0), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        else:
            ax.imshow(clay_c, interpolation="lanczos")
        fig.text((x + cw / 2) / W, (yc - hdr - tile - 1.1 * mm) / H, txt, ha="center", va="top", fontsize=5.8,
                 color=INK, linespacing=1.3)
        if k < 3:
            ya = (yc - hdr - tile / 2) / H
            fig.add_artist(FancyArrowPatch(((x + cw + 0.6 * mm) / W, ya), ((x + cw + ga - 0.6 * mm) / W, ya),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                           color=INK2, lw=0.9))
    # objective + phase-dependent prior, under the stages it acts on
    yo = yc - ch - obj_gap
    xo1 = W
    card(xb, yo, xo1 - xb, obj_h, fc=CARD)
    fig.text((xb + 2.0 * mm) / W, (yo - 2.9 * mm) / H,
             "Loss = photometric (I, Î) + silhouette (night side excluded) + multi-view NCC + normal consistency;"
             " by phase angle:", fontsize=6.3, va="center", ha="left", color=INK)
    chip_w, chip_h = (xo1 - xb - 6.0 * mm) / 2, 3.4 * mm
    for i, (txt, (c, fc)) in enumerate((("Phase < 15°: albedo total variation", TV),
                                        ("Phase ≥ 15°: cast shadows in V (Sun-view depth)", SHADOW))):
        x = xb + 2.0 * mm + i * (chip_w + 2.0 * mm)
        card(x, yo - 5.4 * mm, chip_w, chip_h, fc=fc, r=0.8 * mm)
        fig.add_artist(matplotlib.lines.Line2D([(x + 0.9 * mm) / W] * 2, [(yo - 5.4 * mm - 0.6 * mm) / H,
                                               (yo - 5.4 * mm - chip_h + 0.6 * mm) / H], color=c, lw=2.2,
                                               solid_capstyle="butt"))
        fig.text((x + chip_w / 2) / W, (yo - 5.4 * mm - chip_h / 2) / H, txt, fontsize=5.8, va="center", ha="center",
                 color=INK)
    # gradients flow back from the objective to the surfels
    xg = xb + (cw + ga) + cw / 2
    fig.add_artist(FancyArrowPatch((xg / W, yo / H), (xg / W, (yc - ch - 0.3 * mm) / H), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, color=ACC, lw=0.9))
    fig.text((xg + 1.0 * mm) / W, (yo + obj_gap / 2) / H, "gradients to the surfels", fontsize=5.5, va="center",
             ha="left", color=INK)

    # ---- c-h: decomposition of a held-out image, tiled on black like Figs. 2 and 4
    rt = H - top_h - 5.0 * mm
    res = z["pred"] - z["gt"]          # render minus image, as in Fig. 10
    alb, phi = z["albedo"][sy, sx][m], z["phi"][sy, sx][m]
    panels = [("Observed $I$", np.where(m, z["gt"][sy, sx], 0), dict(cmap="gray", vmin=0, vmax=1)),
              ("Rendered $\\hat{I}$", np.where(m, z["pred"][sy, sx], 0), dict(cmap="gray", vmin=0, vmax=1)),
              ("Albedo $A$", np.where(m, z["albedo"][sy, sx], np.nan),
               dict(cmap="gray", vmin=np.percentile(alb, 2), vmax=np.percentile(alb, 98))),
              ("Shading $\\Phi$", np.where(m, z["phi"][sy, sx], np.nan),
               dict(cmap="gray", vmin=np.percentile(phi, 1), vmax=np.percentile(phi, 99.5))),
              ("Normals $n$", None, None),
              ("Residual $\\hat{I} - I$", np.where(m, res[sy, sx], np.nan), dict(cmap="PuOr_r", vmin=-0.15, vmax=0.15))]
    fig.text(0, (rt + 0.4 * mm) / H, "c", fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)
    fig.text(3.4 * mm / W, (rt + 0.55 * mm) / H,
             f"Test image of Ryugu decomposed by the trained model (ONC-T, phase {meta['image']['phase']:.0f}°)",
             fontsize=6.5, va="bottom", ha="left", color=INK)
    for k, (t, img, kw) in enumerate(panels):
        x = k * (pw + g)
        fig.text((x + pw / 2) / W, (rt - head / 2) / H, t, ha="center", va="center", fontsize=6.4, color=INK)
        ax = ax_at(x, rt - head, pw, pw)
        if img is None:
            nrm = z["normal"][:, sy, sx]
            rgb = np.stack([0.5 + 0.5 * nrm[0], 0.5 - 0.5 * nrm[1], 0.5 - 0.5 * nrm[2]], -1)
            rgb[~m] = 0
            ax.imshow(np.clip(rgb, 0, 1), interpolation="lanczos")
        else:
            cm = matplotlib.colormaps[kw.pop("cmap")].copy()
            cm.set_bad("black")
            ax.imshow(img, cmap=cm, interpolation="lanczos" if k < 5 else "nearest", **kw)
    rmse = float(np.sqrt(np.mean(res[sy, sx][m] ** 2)))
    rf.tile_text(ax, 0.5, 0.025, f"RMS {rmse:.3f}", size=5.5, ha="center", va="bottom")
    cax = fig.add_axes([(W - cb_w + 1.4 * mm) / W, (rt - head - 0.9 * pw) / H, 1.3 * mm / W, 0.8 * pw / H])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(-0.15, 0.15), "PuOr_r"), cax=cax,
                      orientation="vertical")
    cb.set_ticks([-0.15, 0, 0.15])
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Î − I", fontsize=6, labelpad=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
    print("wrote", out, f"{H / mm:.0f} mm high")


if __name__ == "__main__":
    main()
