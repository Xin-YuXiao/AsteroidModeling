"""Render panels for the A&A figures (CPU): per body, the reference and the meshes of
PCGS (final configuration), PGSR and Asteroid-GS from the camera of one held-out image of split 0,
as clay shading, a zoom window and the signed distance to the reference (nc_fig_recon.render, with
the final PCGS run, Bennu added and finished panels skipped). Then 8-bit PNG copies for composing
on another machine (aa_figs.py).

usage: python aa_render.py <panel dir> [body ...]     (default: itokawa ryugu bennu)
       python aa_render.py <panel dir> --png           (write <name>.png next to every <name>.npy)
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nc_fig_recon as nr  # noqa: E402

SPLIT = 0
PCGS = "pcgs-v18n"
METHODS = {
    "itokawa": [("ref", None), ("pcgs", PCGS + "_s{s}/mesh_mvpoisson_fairoff"), ("pgsr", "pgsr_s{s}/mesh_pgsr"),
                ("ags", "ags_s{s}/mesh_ags")],
    "ryugu": [("ref", None), ("pcgs", PCGS + "_s{s}/mesh_mvpoisson_fairoff"), ("pgsr", "pgsr_s{s}/mesh_pgsr"),
              ("ags", "ags_s{s}/mesh_ags")],
    # Asteroid-GS ran on Bennu only with the scene extent of its repository
    "bennu": [("ref", None), ("pcgs", PCGS + "_s{s}/mesh_mvpoisson_fairoff"), ("pgsr", "pgsr_s{s}/mesh_pgsr"),
              ("ags", "ags-camext_s{s}/mesh_ags")],
}
TEST = {"itokawa": "ST_2421076883_v.png", "ryugu": None, "bennu": None}


def pick_test(raycast, root):
    """The held-out image of split 0 whose body mask is largest without touching the frame border."""
    best = None
    for c in raycast.load_cameras(f"{root}/scene_split{SPLIT}", "test"):
        m = c["mask"]
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        if best is None or m.sum() > best["mask"].sum():
            best = c
    return best


def render(pdir, bodies):
    import paper_figs as pf
    import raycast
    import render_hq as hq
    pf.SUFFIX, pf.THREADS = "_sfm_t20", int(os.environ.get("THREADS", 8))
    hq.CLAY[:] = (0.60, 0.585, 0.56)
    os.makedirs(pdir, exist_ok=True)
    mpath = f"{pdir}/meta.json"
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for body in bodies:
        ref = pf.load_ref(body)
        root = pf.D + f"/data/img/{body}"
        if TEST[body]:
            cam = next(c for c in raycast.load_cameras(f"{root}/scene_split{SPLIT}", "test") if c["name"] == TEST[body])
        elif body == "ryugu":
            names = sorted(l.strip() for l in open(f"{root}/scene_split{SPLIT}/test_images.txt") if l.strip())
            cam = next(c for c in raycast.load_cameras(f"{root}/scene_split{SPLIT}", "test") if c["name"] == names[0])
        else:
            cam = pick_test(raycast, root)
        ys, xs = np.nonzero(cam["mask"])
        rot = bool(ys.ptp() > 1.2 * xs.ptp())
        T = pf.frame_tf(body)[:3, :3]
        Rb = cam["R"] @ np.linalg.inv(T / np.cbrt(np.linalg.det(T)))
        d, up = -Rb[2], (Rb[0] if rot else -Rb[1])
        half = 1.03 * float(np.linalg.norm(ref.vertices, axis=1).max())
        Tf = pf.frame_tf(body)
        Xc = cam["R"] @ (np.linalg.solve(Tf, np.array([0.0, 0, 0, 1]))[:3] - cam["C"])
        px_m = float(np.cbrt(np.linalg.det(Tf[:3, :3])) * Xc[2] * 1000 / cam["fx"])
        mb = meta.setdefault(body, dict(half=half, res=1500, rot=rot, stats={}))
        for key, spec in METHODS[body]:
            done = all(os.path.exists(f"{pdir}/{body}_{key}_{k}.npy") for k in (("clay", "zoom") if spec is None else
                                                                                    ("clay", "zoom", "err")))
            if done and (spec is None or key in mb["stats"]):
                print(body, key, "exists", flush=True)
                continue
            m = ref if spec is None else pf.load_rec(body, spec.format(s=SPLIT))
            r = hq.Renderer(m.vertices, m.faces, pf.THREADS)
            clay = r.render(d, up, half, 1500, **nr.LIGHT)
            np.save(f"{pdir}/{body}_{key}_clay.npy", clay.astype(np.float16))
            if key == "ref":
                mb["zoom"] = nr.zoom_window(clay, half, nr.ZOOM_PX * px_m)
            zx, zy, zw = mb["zoom"]
            np.save(f"{pdir}/{body}_{key}_zoom.npy",
                    r.render(d, up, half, 1000, window=(zx, zy, zw), **nr.LIGHT).astype(np.float16))
            if spec is not None:
                sd = pf.signed_dist(ref, m.vertices)
                mb["stats"][key] = dict(mean_abs=float(np.abs(sd).mean()), rmse=float(np.sqrt((sd ** 2).mean())),
                                        over3=float((np.abs(sd) > nr.VMAX).mean()), spec=spec.format(s=SPLIT))
                np.save(f"{pdir}/{body}_{key}_sd_hist.npy", np.histogram(np.clip(sd, -6, 6), 240, (-6, 6))[0])
                img = r.render(d, up, half, 1500, colors=hq.diverging(sd, nr.VMAX), **nr.ERR_LIGHT)
                np.save(f"{pdir}/{body}_{key}_err.npy", img.astype(np.float16))
            print(body, key, flush=True)
            json.dump(meta, open(mpath, "w"), indent=1)
            del r, m
        np.save(f"{pdir}/{body}_image.npy", (cam["img"] * 255).astype(np.uint8))
        mb["image"] = dict(name=cam["name"], utc=cam["utc"], phase=cam["phase"],
                           u0=float(cam["fx"] * Xc[0] / Xc[2] + cam["cx"] - 0.5),
                           v0=float(cam["fy"] * Xc[1] / Xc[2] + cam["cy"] - 0.5), px_m=px_m)
        json.dump(meta, open(mpath, "w"), indent=1)


def to_png(pdir):
    import cv2
    for f in sorted(glob.glob(f"{pdir}/*.npy")):
        a = np.load(f)
        if a.ndim == 1:
            continue
        a = a.astype(np.float32)
        if a.max() > 1.5:
            a = a / 255
        a = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
        if a.ndim == 3 and a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA)
        elif a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f[:-4] + ".png", a)
    print("png done", flush=True)


if __name__ == "__main__":
    if "--png" in sys.argv:
        to_png(sys.argv[1])
    else:
        render(sys.argv[1], sys.argv[2:] or ["itokawa", "ryugu", "bennu"])
