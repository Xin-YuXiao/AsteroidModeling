# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""SPICE geometry for every AMICA v-band frame in the Asteroid-GS window, plus a
one-off check of the FITS pixel orientation against the reference silhouette.

Outputs ($PCGS_DATA/itokawa/):
  spice_geometry.json  per frame: et, UTC, camera-to-body rotation, camera centre
                       and Sun direction in ITOKAWA_FIXED (km), distance, phase
  ref_mesh_km.ply      Gaskell SPC reference (3.1M facets), welded
The reference mesh is used only to fix the global image orientation convention
(flip/transpose) and to measure SPICE pointing error; it never enters any pose
or reconstruction that is evaluated.
Step 1 of 3 for Itokawa (then data/itokawa_prep.py, data/itokawa_refine.py); needs the kernels,
FITS frames and Gaskell shape of data/fetch_itokawa.sh.
usage: python data/itokawa_spice.py
"""
import glob
import json
import os

import numpy as np
import open3d as o3d
import spiceypy as sp
import trimesh
from astropy.io import fits

ROOT = os.path.abspath(os.environ.get(
    "PCGS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")))
D = os.path.join(ROOT, "itokawa")
K = os.path.join(D, "spice")
KERNELS = ["naif0012.tls", "hayabusa.tsc", "de440s.bsp", "itokawa_1989-2010.bsp",
           "hayabusa_itokawarendezvous_v01.bsp", "itokawa_gaskell.tpc", "itokawa_fixed.tf",
           "HAYABUSA_HP.TF", "amica31.ti", "hayabusa_itokawarendezvous_v01.bc"]
F_PX = 0.1208 / 0.000012          # focal length / pixel pitch (amica31.ti)
C_PX = 511.5


def load_ref():
    out = os.path.join(D, "ref_mesh_km.ply")
    if os.path.exists(out):
        return trimesh.load(out, process=False)
    tri = np.loadtxt(os.path.join(D, "shape/itokawa_f3145728.tri"), usecols=range(9))
    m = trimesh.Trimesh(vertices=tri.reshape(-1, 3), faces=np.arange(len(tri) * 3).reshape(-1, 3),
                        process=False)
    m.merge_vertices(digits_vertex=6)
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    m.export(out)
    return m


def main():
    for k in KERNELS:
        sp.furnsh(os.path.join(K, k))
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "fits", "ST_*_v.fits"))):
        h = fits.getheader(f)
        utc = str(h["UTC_0"]).rstrip(".")
        exp = float(h["EXP_0"])
        et_utc = sp.str2et(utc)
        try:
            et_clk = sp.scs2e(-130, str(h["TI_0"]))
        except Exception:  # noqa: BLE001
            et_clk = float("nan")
        et = et_utc + 0.5 * exp
        try:
            # spkpos(target, et, frame, abcorr, observer) is the observer->target
            # vector, i.e. the camera centre in the body frame (evaluated at et-lt)
            scpos, lt = sp.spkpos("HAYABUSA", et, "ITOKAWA_FIXED", "CN+S", "ITOKAWA")
            r_cam2j = sp.pxform("HAYABUSA_AMICA", "J2000", et)
            r_j2body = sp.pxform("J2000", "ITOKAWA_FIXED", et - lt)
            sun, _ = sp.spkpos("SUN", et - lt, "ITOKAWA_FIXED", "LT+S", "ITOKAWA")
        except Exception as e:  # noqa: BLE001
            rows.append(dict(file=os.path.basename(f), error=str(e)[:120]))
            continue
        C = np.asarray(scpos)
        R = np.asarray(r_j2body) @ np.asarray(r_cam2j)       # camera -> body
        s = np.asarray(sun) / np.linalg.norm(sun)
        v = -C / np.linalg.norm(C)                           # camera -> body centre
        rows.append(dict(
            file=os.path.basename(f), utc=utc, et=et, et_clock_minus_utc=et_clk - et_utc,
            exposure=exp, cam_center_km=C.tolist(), R_cam2body=R.tolist(),
            sun_dir_body=s.tolist(), distance_km=float(np.linalg.norm(C)),
            phase_deg=float(np.degrees(np.arccos(np.clip(np.dot(-v, s), -1, 1)))),
            boresight_offset_deg=float(np.degrees(np.arccos(np.clip(R[:, 2] @ v, -1, 1)))),
            subsc_lat_deg=float(np.degrees(np.arcsin(C[2] / np.linalg.norm(C)))),
            subsc_lon_deg=float(np.degrees(np.arctan2(C[1], C[0])))))
    ok = [r for r in rows if "error" not in r]
    print(f"frames {len(rows)}, with geometry {len(ok)}")
    d = np.array([r["distance_km"] for r in ok])
    print("distance km: min %.2f max %.2f, in [7,8]: %d" % (d.min(), d.max(), ((d >= 7) & (d <= 8)).sum()))
    print("boresight offset deg: median %.3f max %.3f" % (
        np.median([r["boresight_offset_deg"] for r in ok]), max(r["boresight_offset_deg"] for r in ok)))
    print("clock-vs-UTC et diff s: median %.3f" % np.nanmedian([r["et_clock_minus_utc"] for r in ok]))
    print("phase deg: %.1f..%.1f  subsc lat: %.1f..%.1f" % (
        min(r["phase_deg"] for r in ok), max(r["phase_deg"] for r in ok),
        min(r["subsc_lat_deg"] for r in ok), max(r["subsc_lat_deg"] for r in ok)))
    json.dump(rows, open(os.path.join(D, "spice_geometry.json"), "w"), indent=1)

    # --- orientation convention: silhouette IoU of the projected reference
    ref = load_ref()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(ref.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(ref.faces, dtype=np.uint32))))
    uu, vv = np.meshgrid(np.arange(1024), np.arange(1024))
    dirs_cam = np.stack([(uu - C_PX) / F_PX, (vv - C_PX) / F_PX, np.ones_like(uu, float)], -1).reshape(-1, 3)
    variants = {"id": lambda a: a, "flipud": np.flipud, "fliplr": np.fliplr, "rot180": lambda a: a[::-1, ::-1],
                "T": lambda a: a.T, "T_flipud": lambda a: np.flipud(a.T), "T_fliplr": lambda a: np.fliplr(a.T),
                "T_rot180": lambda a: a.T[::-1, ::-1]}
    pick = [r for r in ok if 7 <= r["distance_km"] <= 8][::25][:8]
    scores = {k: [] for k in variants}
    for r in pick:
        R, C = np.array(r["R_cam2body"]), np.array(r["cam_center_km"])
        dw = dirs_cam @ R.T
        rays = np.concatenate([np.repeat(C[None], len(dw), 0), dw / np.linalg.norm(dw, axis=1, keepdims=True)], 1)
        hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()).reshape(1024, 1024)
        img = fits.getdata(os.path.join(D, "fits", r["file"])).astype(float)
        m = img > 8
        for k, fn in variants.items():
            mv = fn(m)
            scores[k].append(float((mv & hit).sum() / max((mv | hit).sum(), 1)))
    for k, v in sorted(scores.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"orientation {k:9s} IoU mean {np.mean(v):.3f}  {np.round(v, 3).tolist()}")


if __name__ == "__main__":
    main()
