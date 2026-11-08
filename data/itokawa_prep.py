# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Build the Itokawa image-domain benchmark scene (Asteroid-GS protocol).

Selection follows Zhang+2025 (A&A 704 A247, Sect. 4.1): AMICA v-band frames from
2005-09-28..10-02 at 7-8 km, 60 images, random 50/10 train/test. Their exact
list is unpublished, so we pick 60 frames spread evenly in sub-spacecraft
longitude (rotation coverage) among frames that are 8-bit and fully contain the
body, and publish the list.

Poses: SPICE pointing is off by ~30 px (median, ~20 m at the surface), so we run
SfM (SIFT, fixed AMICA intrinsics) and similarity-align it to the SPICE camera
centres, which puts the model in ITOKAWA_FIXED, km. This mirrors Asteroid-GS
(SfM poses in the body-fixed frame) without using the reference shape.

Writes $PCGS_DATA/itokawa/scene/: images/, masks/, sparse/0/*.txt (COLMAP text,
PINHOLE), meta.json (per-image SPICE geometry incl. Sun direction), splits/.
Step 2 of 3 for Itokawa: data/itokawa_spice.py, this script, data/itokawa_refine.py.
usage: python data/itokawa_prep.py
"""
import json
import os
import shutil

import cv2
import numpy as np
import pycolmap
from astropy.io import fits
from scipy import ndimage

ROOT = os.path.abspath(os.environ.get(
    "PCGS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")))
D = os.path.join(ROOT, "itokawa")
OUT = os.path.join(D, "scene")
N_IMG, N_TEST, SPLIT_SEEDS = 60, 10, [0, 1, 2]
F_PX, C_PX, W = 0.1208 / 0.000012, 511.5, 1024
MASK_DN = 8


def body_mask(img):
    m = img > MASK_DN
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    m = lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    return ndimage.binary_fill_holes(m)


def select(rows):
    cand = []
    for r in rows:
        if "error" in r or not 7.0 <= r["distance_km"] <= 8.0:
            continue
        img = fits.getdata(os.path.join(D, "fits", r["file"]))
        if img.dtype != np.uint8 or img.shape != (W, W):
            continue
        m = body_mask(img)
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m.mean() < 0.02:
            continue
        cand.append(r)
    lon = np.array([r["subsc_lon_deg"] for r in cand])
    picked = []
    for t in np.linspace(-180, 180, N_IMG, endpoint=False):
        dl = np.abs((lon - t + 180) % 360 - 180)
        for i in np.argsort(dl):
            if i not in picked:
                picked.append(int(i))
                break
    print(f"candidates {len(cand)} -> selected {len(picked)}")
    return [cand[i] for i in sorted(picked, key=lambda i: cand[i]["et"])]


def qvec(R):
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[:] = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q) * np.sign(q[0] if q[0] != 0 else 1)


def pose(img):
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    return np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation)


def write_colmap_text(rec, sparse_dir, names):
    os.makedirs(sparse_dir, exist_ok=True)
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {W} {F_PX:.6f} {F_PX:.6f} {C_PX} {C_PX}\n")
    pts = rec.points3D
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        for i, img in enumerate(sorted(rec.images.values(), key=lambda im: im.name), start=1):
            R, t = pose(img)
            q = qvec(R)
            f.write(f"{i} {' '.join(f'{v:.12f}' for v in q)} {' '.join(f'{v:.12f}' for v in t)} 1 {img.name}\n\n")
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        for pid, p in pts.items():
            x = p.xyz
            c = np.clip(np.asarray(p.color), 0, 255).astype(int)
            f.write(f"{pid} {x[0]:.9f} {x[1]:.9f} {x[2]:.9f} {c[0]} {c[1]} {c[2]} {p.error:.4f}\n")


def main():
    rows = json.load(open(os.path.join(D, "spice_geometry.json")))
    sel = select(rows)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sub in ["images", "masks", "sparse/0", "splits", "colmap"]:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    meta = {}
    for r in sel:
        name = r["file"].replace(".fits", ".png")
        img = fits.getdata(os.path.join(D, "fits", r["file"]))
        cv2.imwrite(os.path.join(OUT, "images", name), np.repeat(img[..., None], 3, -1))
        m = (body_mask(img) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT, "masks", name), m)
        # COLMAP mask convention: <mask_path>/<name>.png, 0 = ignore
        cv2.imwrite(os.path.join(OUT, "colmap", name + ".png"),
                    ndimage.binary_dilation(m > 0, iterations=15).astype(np.uint8) * 255)
        meta[name] = {k: r[k] for k in ["file", "utc", "et", "cam_center_km", "R_cam2body", "sun_dir_body",
                                        "distance_km", "phase_deg", "subsc_lat_deg", "subsc_lon_deg"]}
    names = sorted(meta)

    # ---- SfM with fixed AMICA intrinsics
    db = os.path.join(OUT, "colmap", "database.db")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{F_PX},{F_PX},{C_PX},{C_PX}"
    reader.mask_path = os.path.join(OUT, "colmap")
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 16000
    ext.sift.peak_threshold = 0.002  # weak regolith texture
    ext.use_gpu = False
    pycolmap.extract_features(db, os.path.join(OUT, "images"), image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    pycolmap.match_exhaustive(db)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, os.path.join(OUT, "images"), os.path.join(OUT, "colmap"), opts)
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print("SfM:", rec.summary())

    # ---- Sim(3) to SPICE camera centres (body-fixed, km)
    reg = [im.name for im in rec.images.values()]
    tgt = np.array([meta[n]["cam_center_km"] for n in reg])
    ro = pycolmap.RANSACOptions()
    ro.max_error = 0.05  # km
    sim = pycolmap.align_reconstruction_to_locations(rec, reg, tgt, 3, ro)
    rec.transform(sim)
    print("Sim3 scale", sim.scale)

    # ---- pose diagnostics vs SPICE
    diag = {}
    for im in rec.images.values():
        R, t = pose(im)
        C = -R.T @ t
        Rs = np.array(meta[im.name]["R_cam2body"])
        dR = Rs.T @ R.T
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        diag[im.name] = dict(center_diff_m=float(np.linalg.norm(C - meta[im.name]["cam_center_km"]) * 1000),
                             rot_diff_deg=float(ang))
        meta[im.name]["sfm_cam_center_km"] = C.tolist()
        meta[im.name]["sfm_R_cam2body"] = R.T.tolist()
    cd = np.array([v["center_diff_m"] for v in diag.values()])
    rd = np.array([v["rot_diff_deg"] for v in diag.values()])
    print(f"registered {len(diag)}/{len(names)}; |C_sfm - C_spice| median {np.median(cd):.1f} m "
          f"max {cd.max():.1f} m; rot diff median {np.median(rd):.3f} deg max {rd.max():.3f} deg")

    write_colmap_text(rec, os.path.join(OUT, "sparse/0"), names)
    unreg = sorted(set(names) - set(reg))
    rng_names = sorted(reg)
    for s in SPLIT_SEEDS:
        rng = np.random.default_rng(s)
        test = sorted(rng.choice(rng_names, N_TEST, replace=False).tolist())
        open(os.path.join(OUT, "splits", f"test_split{s}.txt"), "w").write("\n".join(test) + "\n")
    json.dump(dict(images=meta, pose_diag=diag, unregistered=unreg,
                   sim3_scale=float(sim.scale), selection_rule="7-8 km, uint8, full body, "
                   "60 evenly spaced in sub-spacecraft longitude"), open(os.path.join(OUT, "meta.json"), "w"),
              indent=1)
    print("unregistered:", unreg)
    print("PREP_DONE")


if __name__ == "__main__":
    main()
