# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""Body-generic builder for the image-domain benchmark (Asteroid-GS protocol).

  python data/build_scene.py <campaign> geometry   SPICE geometry per frame + pixel-orientation test
  python data/build_scene.py <campaign> prep       select 60 frames, export PNG/masks, free SfM
  python data/build_scene.py <campaign> refine     SPICE-initialised triangulation + BA, splits
  python data/build_scene.py <campaign> splits     scene_split{k}/ folders from scene/splits/ (Itokawa)
  python data/build_scene.py <campaign> register   evaluation-only SfM frame -> reference transform
campaigns: itokawa, ryugu_7deg, ryugu_19deg, ryugu_40deg, bennu; data under $PCGS_DATA (default
<repository>/datasets), one folder per campaign (see data/fetch_*.sh).

Selection follows Zhang+2025 (A&A 704 A247, Sect. 4.1): the published time
window and range interval, 60 frames, random 50/10 splits; frames are spread
evenly in sub-spacecraft longitude among those that fully contain the body.
Free incremental SfM on these narrow-FOV frames can converge to the
Necker-reversed solution, so the final poses come from SPICE-initialised
triangulation + bundle adjustment, similarity-aligned to the SPICE camera
centres (body-fixed frame, km). The reference shape is used only to fix the
global pixel-orientation convention and to quantify SPICE pointing error.
(itokawa_spice.py / itokawa_prep.py / itokawa_refine.py are the Itokawa-only
originals this generalises; the Itokawa scene of the paper was built with them.)
"""
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
from scipy import ndimage

ROOT = os.path.abspath(os.environ.get(
    "PCGS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")))
N_IMG, N_TEST, SPLIT_SEEDS = 60, 10, [0, 1, 2]
N_THREADS = int(os.environ.get("OMP_NUM_THREADS", 8))   # COLMAP otherwise uses every core of the shared node
ORIENT = {"id": lambda a: a, "flipud": np.flipud, "fliplr": np.fliplr, "rot180": lambda a: a[::-1, ::-1],
          # OCAMS FITS: rows run along camera +X and columns along +Y (orx_ocams_v07.ti, SPOC_FITS_NAXIS)
          "transpose": lambda a: np.ascontiguousarray(a.T), "antitranspose": lambda a: np.ascontiguousarray(a[::-1, ::-1].T)}


def read_amica(path):
    from astropy.io import fits
    h = fits.open(path)[0]
    img = h.data
    ok = img.dtype == np.uint8 and img.shape == (1024, 1024)
    return img.astype(np.float32), str(h.header["UTC_0"]).rstrip("."), 0.5 * float(h.header["EXP_0"]), ok


def read_onc(path):
    from astropy.io import fits
    h = fits.open(path)[1]
    return h.data.astype(np.float32), str(h.header["DATE-OBS"]), 0.0, h.data.shape == (1024, 1024)


def read_ocams(path):
    """OSIRIS-REx OCAMS L2 I/F frame; DATE_OBS is the exposure start. Long exposures (4 s satellite
    searches, the body saturated) are flagged as unusable."""
    from astropy.io import fits
    h = fits.open(path)[0]
    exp = float(h.header["EXPOSEC"])
    img = np.nan_to_num(h.data.astype(np.float32))
    return img, str(h.header["DATE_OBS"]), 0.5 * exp, img.shape == (1024, 1024) and exp < 0.1


BODIES = {
    "itokawa": dict(
        frames="fits/ST_*_v.fits", reader=read_amica, observer="HAYABUSA", target="ITOKAWA",
        body_frame="ITOKAWA_FIXED", cam_frame="HAYABUSA_AMICA", f_px=0.1208 / 0.000012, c_px=511.5,
        dist_km=(7.0, 8.0), utc_window=None, scale="dn", mask_frac=8 / 255,
        kernels=["naif0012.tls", "hayabusa.tsc", "de440s.bsp", "itokawa_1989-2010.bsp",
                 "hayabusa_itokawarendezvous_v01.bsp", "itokawa_gaskell.tpc", "itokawa_fixed.tf",
                 "HAYABUSA_HP.TF", "amica31.ti", "hayabusa_itokawarendezvous_v01.bc"]),
    "ryugu_19deg": dict(
        frames="raw/20180710/hyb2_onc_*_tvf_l2c.fit", reader=read_onc, observer="HAYABUSA2", target="RYUGU",
        body_frame="RYUGU_FIXED", cam_frame="HAYABUSA2_ONC-T", f_px=120.50 / 0.013, c_px=511.5,
        dist_km=(19.5, 21.5), utc_window=("2018-07-10T06:05", "2018-07-10T13:55"), scale="p999",
        mask_frac=0.04,
        kernels=["naif0012.tls", "pck00010.tpc", "hyb2_ryugu_shape_v20190328.tpc", "hyb2_v14.tf",
                 "hyb2_hp_v01.tf", "hyb2_ryugu_v01.tf", "hyb2_onc_v05.ti", "hyb2_20141203-20191231_v01.tsc",
                 "de440s.bsp", "2162173_Ryugu.bsp", "2162173_ryugu_20180601-20191230_0060_20181221.bsp",
                 "hyb2_approach_od_v20180811114238.bsp", "hyb2_hpk_20180627_20190213_v01.bsp",
                 "lidar_derived_trj_20191114_20180630053224_20190213030000_v02.bsp",
                 "hyb2_hkattrpt_2018_v02.bc", "hyb2_aocsc_2018_v02.bc",
                 "hyb2_hkattrpt_20190101000000_20190213030000_v02.bc",
                 "hyb2_aocsc_20190101000000_20190213030000_v02.bc"]),
    # OSIRIS-REx MapCam, Approach + Preliminary Survey (Nov-Dec 2018), 60 frames pre-selected from
    # SPICE geometry alone by data/select_bennu.py (phase < 60 deg, >= 60% of the body in the 4 deg
    # FOV, so frames may cut the body); reference: OLA altimetry-only global model v21 (0.88 m),
    # independent of the images. MapCam has no distortion model in the IK (orx_ocams_v07.ti).
    "bennu": dict(
        frames="frames/*_map_iofL2pan.fits", reader=read_ocams, observer="-64", target="BENNU",
        body_frame="IAU_BENNU", cam_frame="ORX_OCAMS_MAPCAM", f_px=125.20 / 0.0085, c_px=511.5,
        dist_km=(5.0, 45.0), utc_window=None, scale="p999", mask_frac=0.04, preselected=True,
        kernels=["naif0012.tls", "orx_sclkscet_00093.tsc", "orx_v14.tf", "orx_ocams_v07.ti", "bennu_v17.tpc",
                 "de424.bsp", "bennu_refdrmc_v1.bsp", "orx_180801_190302_181218_od077_v1.bsp",
                 "orx_181203_190302_190104_od085_v1.bsp", "orx_struct_v04.bsp",
                 "orx_sc_rel_181126_181202_v01.bc", "orx_sc_rel_181203_181209_v01.bc",
                 "orx_sc_rel_181210_181216_v01.bc", "orx_struct_mapcam_v01.bc"]),
}


# Real-data campaigns at other phase angles / latitudes (same camera, kernels and reference
# as the base body; directories symlink spice/, shape/, ref_mesh_km.ply, orientation.json).
CAMPAIGNS = {
    "ryugu_40deg": ("ryugu_19deg", dict(frames="raw/20180831/hyb2_onc_*_tvf_l2c.fit", dist_km=(19.0, 25.0), utc_window=None,
                                        n_img=48, n_test=8)),          # BOX-B2 global mapping at ~40 deg phase
    "ryugu_7deg": ("ryugu_19deg", dict(frames="raw/20190115/hyb2_onc_*_tvf_l2c.fit", dist_km=(18.0, 23.0), utc_window=None)),  # BOX-A 2019-01-15, ~7.2 deg (pre-TD1)
}
for _name, (_base, _over) in CAMPAIGNS.items():
    BODIES[_name] = {**BODIES[_base], **_over, "skip_orient": True}


def body_mask(img, thr):
    m = img > thr
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    m = lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    return ndimage.binary_fill_holes(m)


def load_ref(d):
    import trimesh
    out = os.path.join(d, "ref_mesh_km.ply")
    if os.path.exists(out):
        return trimesh.load(out, process=False)
    src = sorted(glob.glob(os.path.join(d, "shape", "*")), key=os.path.getsize)[-1]
    if src.endswith(".tri"):
        tri = np.loadtxt(src, usecols=range(9))
        m = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(len(tri) * 3).reshape(-1, 3), process=False)
    else:
        m = trimesh.load(src, force="mesh", process=False)
    m.merge_vertices(digits_vertex=6)
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    m.export(out)
    print(f"reference {os.path.basename(src)}: {len(m.faces)} faces, watertight={m.is_watertight}, "
          f"volume {m.volume:.5f} km3")
    return m


# ---------------------------------------------------------------- geometry
def geometry(body):
    import open3d as o3d
    import spiceypy as sp
    cfg, d = BODIES[body], os.path.join(ROOT, body)
    for k in cfg["kernels"]:
        sp.furnsh(os.path.join(d, "spice", k))
    rows = []
    for f in sorted(glob.glob(os.path.join(d, cfg["frames"]))):
        img, utc, dt, ok = cfg["reader"](f)
        if cfg["utc_window"] and not cfg["utc_window"][0] <= utc[:16] <= cfg["utc_window"][1]:
            continue
        et = sp.str2et(utc) + dt
        try:
            C, lt = sp.spkpos(cfg["observer"], et, cfg["body_frame"], "CN+S", cfg["target"])
            R = np.asarray(sp.pxform("J2000", cfg["body_frame"], et - lt)) @ \
                np.asarray(sp.pxform(cfg["cam_frame"], "J2000", et))
            sun = np.asarray(sp.spkpos("SUN", et - lt, cfg["body_frame"], "LT+S", cfg["target"])[0])
        except Exception as e:  # noqa: BLE001
            rows.append(dict(file=os.path.relpath(f, d), error=str(e)[:160]))
            continue
        C = np.asarray(C)
        s = sun / np.linalg.norm(sun)
        rows.append(dict(
            file=os.path.relpath(f, d), utc=utc, et=et, ok_format=bool(ok), cam_center_km=C.tolist(),
            R_cam2body=R.tolist(), sun_dir_body=s.tolist(), distance_km=float(np.linalg.norm(C)),
            phase_deg=float(np.degrees(np.arccos(np.clip(np.dot(C / np.linalg.norm(C), s), -1, 1)))),
            subsc_lat_deg=float(np.degrees(np.arcsin(C[2] / np.linalg.norm(C)))),
            subsc_lon_deg=float(np.degrees(np.arctan2(C[1], C[0]))),
            subsolar_lat_deg=float(np.degrees(np.arcsin(s[2])))))
    ok = [r for r in rows if "error" not in r]
    print(f"{body}: frames {len(rows)}, with SPICE geometry {len(ok)}")
    if not ok:
        print([r["error"] for r in rows[:3]])
        return
    for k in ["distance_km", "phase_deg", "subsc_lat_deg", "subsolar_lat_deg"]:
        v = np.array([r[k] for r in ok])
        print(f"  {k}: {v.min():.2f} .. {v.max():.2f}")
    json.dump(rows, open(os.path.join(d, "spice_geometry.json"), "w"), indent=1)
    if cfg.get("skip_orient"):
        return  # campaign of an already-characterised camera: orientation.json is shared

    # pixel orientation: silhouette IoU of the projected reference after a 2D shift search
    ref = load_ref(d)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(ref.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(ref.faces, dtype=np.uint32))))
    uu, vv = np.meshgrid(np.arange(1024), np.arange(1024))
    dc = np.stack([(uu - cfg["c_px"]) / cfg["f_px"], (vv - cfg["c_px"]) / cfg["f_px"],
                   np.ones_like(uu, float)], -1).reshape(-1, 3)
    pick = [r for r in ok if cfg["dist_km"][0] <= r["distance_km"] <= cfg["dist_km"][1]]
    pick = pick[:: max(1, len(pick) // 8)][:8]
    res = {k: [] for k in ORIENT}
    shifts = {k: [] for k in ORIENT}
    for r in pick:
        R, C = np.array(r["R_cam2body"]), np.array(r["cam_center_km"])
        dw = dc @ R.T
        rays = np.concatenate([np.repeat(C[None], len(dw), 0), dw / np.linalg.norm(dw, axis=1, keepdims=True)], 1)
        hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()).reshape(1024, 1024)
        img = cfg["reader"](os.path.join(d, r["file"]))[0]
        thr = cfg["mask_frac"] * (255 if cfg["scale"] == "dn" else np.percentile(img, 99.9))
        for k, fn in ORIENT.items():
            m = fn(body_mask(img, thr))
            best = (0.0, 0, 0)
            for step, rng in [(4, range(-100, 101, 4)), (1, None)]:
                cands = [(a, b) for a in rng for b in rng] if rng else \
                    [(best[1] + a, best[2] + b) for a in range(-4, 5) for b in range(-4, 5)]
                for dx, dy in cands:
                    h = np.roll(np.roll(hit, dy, 0), dx, 1)
                    iou = (h & m).sum() / max((h | m).sum(), 1)
                    if iou > best[0]:
                        best = (iou, dx, dy)
            res[k].append(best[0])
            shifts[k].append(np.hypot(best[1], best[2]))
    for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  orientation {k:7s} shifted-IoU mean {np.mean(v):.3f} min {np.min(v):.3f}")
    best_o = max(res, key=lambda k: np.mean(res[k]))
    json.dump(dict(orientation=best_o, iou={k: float(np.mean(v)) for k, v in res.items()},
                   spice_pointing_shift_px_median=float(np.median(shifts[best_o]))),
              open(os.path.join(d, "orientation.json"), "w"), indent=1)
    print(f"  -> orientation {best_o}; SPICE pointing offset median {np.median(shifts[best_o]):.1f} px")


def allow_long_focal(opts):
    """COLMAP treats cameras with focal length > 10 x the image size as bogus and skips them in
    mapping and triangulation; MapCam is 14.4 x (AMICA 9.8, ONC-T 9.1, just under the limit)."""
    opts.mapper.max_focal_length_ratio = opts.triangulation.max_focal_length_ratio = 100.0


# ---------------------------------------------------------------- prep
def prep(body):
    import pycolmap
    cfg, d = BODIES[body], os.path.join(ROOT, body)
    out = os.path.join(d, "scene")
    orient = ORIENT[json.load(open(os.path.join(d, "orientation.json")))["orientation"]]
    rows = [r for r in json.load(open(os.path.join(d, "spice_geometry.json")))
            if "error" not in r and r["ok_format"] and cfg["dist_km"][0] <= r["distance_km"] <= cfg["dist_km"][1]]
    imgs = {r["file"]: orient(cfg["reader"](os.path.join(d, r["file"]))[0]) for r in rows}
    scale = 255.0 if cfg["scale"] == "dn" else float(np.percentile(np.concatenate(
        [v[v > 0].ravel()[::7] for v in imgs.values()]), 99.9))
    thr = cfg["mask_frac"] * scale
    cand = []
    for r in rows:
        m = body_mask(imgs[r["file"]], thr)
        if cfg.get("preselected") or not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m.mean() < 0.02):
            cand.append(r)
    lon = np.array([r["subsc_lon_deg"] for r in cand])
    n_img = min(cfg.get("n_img", N_IMG), len(cand))
    picked = []
    for t in np.linspace(-180, 180, n_img, endpoint=False):
        for i in np.argsort(np.abs((lon - t + 180) % 360 - 180)):
            if i not in picked:
                picked.append(int(i))
                break
    sel = [cand[i] for i in sorted(picked, key=lambda i: cand[i]["et"])]
    print(f"{body}: in range {len(rows)}, full-body {len(cand)} -> selected {len(sel)}; 8-bit scale {scale:.4g}")
    if os.path.exists(out):
        shutil.rmtree(out)
    for sub in ["images", "masks", "sparse/0", "splits", "colmap"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    meta = {}
    for r in sel:
        name = os.path.splitext(os.path.basename(r["file"]))[0] + ".png"
        img = imgs[r["file"]]
        u8 = np.clip(np.round(img / scale * 255), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out, "images", name), np.repeat(u8[..., None], 3, -1))
        m = body_mask(img, thr)
        cv2.imwrite(os.path.join(out, "masks", name), (m * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out, "colmap", name + ".png"),
                    ndimage.binary_dilation(m, iterations=15).astype(np.uint8) * 255)
        meta[name] = {k: r[k] for k in ["file", "utc", "et", "cam_center_km", "R_cam2body", "sun_dir_body",
                                        "distance_km", "phase_deg", "subsc_lat_deg", "subsc_lon_deg"]}
    names = sorted(meta)
    db = os.path.join(out, "colmap", "database.db")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{cfg['f_px']},{cfg['f_px']},{cfg['c_px']},{cfg['c_px']}"
    reader.mask_path = os.path.join(out, "colmap")
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 16000
    ext.sift.peak_threshold = 0.002
    ext.use_gpu = False
    ext.num_threads = N_THREADS
    pycolmap.extract_features(db, os.path.join(out, "images"), image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext)
    mo = pycolmap.FeatureMatchingOptions()
    mo.use_gpu, mo.num_threads = False, N_THREADS
    pycolmap.match_exhaustive(db, matching_options=mo)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
    allow_long_focal(opts)
    recs = pycolmap.incremental_mapping(db, os.path.join(out, "images"), os.path.join(out, "colmap"), opts)
    if recs:
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        os.makedirs(os.path.join(out, "colmap", "0"), exist_ok=True)
        rec.write(os.path.join(out, "colmap", "0"))
        print(f"free SfM: {rec.num_reg_images()}/{len(names)} registered, {rec.num_points3D()} points, "
              f"{rec.compute_mean_reprojection_error():.3f} px")
    else:  # e.g. Bennu MapCam (4 deg FOV): no initial pair; refine starts from the database + SPICE poses
        print("free SfM: no model (no initial pair); refine uses all database images with SPICE poses")
    json.dump(dict(images=meta, scale_to_u8=scale, mask_threshold=thr, selection_rule=(
        f"{cfg['dist_km']} km, {cfg['utc_window']}, full body, {n_img} evenly spaced in sub-spacecraft longitude")),
        open(os.path.join(out, "meta.json"), "w"), indent=1)


# ---------------------------------------------------------------- refine
def rec_from_database(db_path, meta):
    """Reconstruction holding every database image at its SPICE pose (keypoints as points2D, no
    3D points), for scenes where free incremental SfM finds no initial pair."""
    import pycolmap
    db = pycolmap.Database.open(db_path)
    rec = pycolmap.Reconstruction()
    for c in db.read_all_cameras():
        rec.add_camera(c)
    for r in db.read_all_rigs():
        rec.add_rig(r)
    images = db.read_all_images()
    frames = {f.frame_id: f for f in db.read_all_frames()}
    for im in images:
        R = np.array(meta[im.name]["R_cam2body"]).T
        f = frames[im.frame_id]
        f.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ np.array(meta[im.name]["cam_center_km"]))
        rec.add_frame(f)
    for im in images:
        kp = db.read_keypoints(im.image_id)[:, :2].astype(np.float64)
        im.points2D = [pycolmap.Point2D(p) for p in kp]
        rec.add_image(im)
    for fid in frames:
        rec.register_frame(fid)
    db.close()
    return rec


def triangulate_tracks(rec, db_path, max_px, min_angle_deg=1.0):
    """Multi-view DLT triangulation of the database feature tracks (union of the verified inlier
    matches) at the current poses of `rec`, replacing its points; observations with depth <= 0 or
    reprojection error >= max_px are dropped, and points need >= 2 observations and a triangulation
    angle >= min_angle_deg. Used for scenes built by rec_from_database, where COLMAP's
    triangulator creates no points."""
    import pycolmap
    db = pycolmap.Database.open(db_path)
    pair_ids, tvgs = db.read_two_view_geometries()
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for pid, g in zip(pair_ids, tvgs):
        i1, i2 = pycolmap.pair_id_to_image_pair(pid)
        for a, b in np.asarray(g.inlier_matches):
            ra, rb = find((i1, int(a))), find((i2, int(b)))
            if ra != rb:
                parent[ra] = rb
    tracks = {}
    for node in list(parent):
        tracks.setdefault(find(node), []).append(node)
    kps = {i: db.read_keypoints(i)[:, :2].astype(np.float64) for i in rec.images}
    db.close()
    for pid in list(rec.points3D):
        rec.delete_point3D(pid)
    cams = {}
    for iid, im in rec.images.items():
        T = np.asarray(im.cam_from_world().matrix())
        cams[iid] = (rec.cameras[im.camera_id].calibration_matrix() @ T, T, -T[:, :3].T @ T[:, 3])
    n = 0
    for nodes in tracks.values():
        if len(nodes) < 2 or len({i for i, _ in nodes}) != len(nodes):
            continue                               # single view, or two keypoints of one image
        A = []
        for i, k in nodes:
            P, x = cams[i][0], kps[i][k]
            A += [x[0] * P[2] - P[0], x[1] * P[2] - P[1]]
        Xh = np.linalg.svd(np.asarray(A))[2][-1]
        if abs(Xh[3]) < 1e-12:
            continue
        X = Xh[:3] / Xh[3]
        keep = []
        for i, k in nodes:
            P, T, _ = cams[i]
            if T[2] @ np.append(X, 1) <= 0:
                continue
            p = P @ np.append(X, 1)
            if np.linalg.norm(p[:2] / p[2] - kps[i][k]) < max_px:
                keep.append((i, k))
        if len(keep) < 2:
            continue
        rays = np.array([(X - cams[i][2]) / np.linalg.norm(X - cams[i][2]) for i, _ in keep])
        if np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))) < min_angle_deg:
            continue
        tr = pycolmap.Track()
        for i, k in keep:
            tr.add_element(i, k)
        rec.add_point3D(X, tr)
        n += 1
    return n


def refine(body):
    import pycolmap
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from itokawa_prep import pose, write_colmap_text
    from itokawa_refine import ROUNDS, rel_rot_agreement, set_pose
    import itokawa_prep
    cfg, d = BODIES[body], os.path.join(ROOT, body)
    out = os.path.join(d, "scene")
    itokawa_prep.F_PX, itokawa_prep.C_PX = cfg["f_px"], cfg["c_px"]  # used by write_colmap_text
    db = os.path.join(out, "colmap", "database.db")
    mj = json.load(open(os.path.join(out, "meta.json")))
    meta = mj["images"]
    if os.path.exists(os.path.join(out, "colmap", "0")):
        rec = pycolmap.Reconstruction(os.path.join(out, "colmap", "0"))
        free = (rel_rot_agreement(rec, meta), rel_rot_agreement(rec, meta, np.diag([-1., -1., 1.])))
        print(f"free SfM rel-rot vs SPICE: as-is {free[0]:.3f} deg, Necker-conjugated {free[1]:.3f} deg "
              f"-> {'NECKER-REVERSED' if free[1] < free[0] else 'consistent'}")
    else:
        rec = rec_from_database(db, meta)
        free = (float("nan"), float("nan"))
        print(f"no free SfM model: {len(rec.images)} database images with SPICE poses")
    for im in rec.images.values():
        R = np.array(meta[im.name]["R_cam2body"]).T
        set_pose(rec, im, R, -R @ np.array(meta[im.name]["cam_center_km"]))
    opts = pycolmap.IncrementalPipelineOptions()
    allow_long_focal(opts)
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
    from_db = not os.path.exists(os.path.join(out, "colmap", "0"))
    for k, (tri_px, filt_px) in enumerate(ROUNDS):
        if from_db:
            triangulate_tracks(rec, db, filt_px)
        else:
            opts.triangulation.complete_max_reproj_error = tri_px
            opts.triangulation.merge_max_reproj_error = tri_px
            opts.triangulation.create_max_angle_error = 2.0 * tri_px / 10.0
            opts.mapper.filter_max_reproj_error = filt_px
            rec = pycolmap.triangulate_points(rec, db, os.path.join(out, "images"),
                                              os.path.join(out, "colmap", f"tri{k}"), clear_points=True, options=opts)
        pycolmap.bundle_adjustment(rec, ba)
        print(f"round {k}: points {rec.num_points3D()} reproj {rec.compute_mean_reprojection_error():.3f} px "
              f"rel-rot vs SPICE {rel_rot_agreement(rec, meta):.3f} deg")
    names = [im.name for im in rec.images.values()]
    ro = pycolmap.RANSACOptions()
    ro.max_error = 0.1 * max(1.0, np.mean(cfg["dist_km"]) / 7.5)
    sim = pycolmap.align_reconstruction_to_locations(
        rec, names, np.array([meta[n]["cam_center_km"] for n in names]), 3, ro)
    rec.transform(sim)
    cd, rd = [], []
    for im in rec.images.values():
        R, t = pose(im)
        C = -R.T @ t
        dR = np.array(meta[im.name]["R_cam2body"]).T @ R.T
        cd.append(np.linalg.norm(C - meta[im.name]["cam_center_km"]) * 1000)
        rd.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        meta[im.name]["sfm_cam_center_km"] = C.tolist()
        meta[im.name]["sfm_R_cam2body"] = R.T.tolist()
    print(f"Sim3 scale {sim.scale:.5f}; |C - C_spice| median {np.median(cd):.1f} m; "
          f"pointing correction median {np.median(rd):.3f} deg")
    write_colmap_text(rec, os.path.join(out, "sparse/0"), names)
    for s in SPLIT_SEEDS:
        rng = np.random.default_rng(s)
        test = sorted(rng.choice(sorted(names), cfg.get("n_test", N_TEST), replace=False).tolist())
        open(os.path.join(out, "splits", f"test_split{s}.txt"), "w").write("\n".join(test) + "\n")
        sd = os.path.join(d, f"scene_split{s}")
        os.makedirs(sd, exist_ok=True)
        for sub in ["images", "masks", "sparse"]:
            if not os.path.lexists(os.path.join(sd, sub)):
                os.symlink(os.path.join(out, sub), os.path.join(sd, sub))
        open(os.path.join(sd, "test_images.txt"), "w").write("\n".join(test) + "\n")
    mj["sfm"] = dict(free_sfm_relrot_deg=free[0], free_sfm_relrot_necker_deg=free[1],
                     method="SPICE-initialised triangulation + 3 rounds BA, Sim3 to SPICE centres",
                     center_diff_m_median=float(np.median(cd)), pointing_corr_deg_median=float(np.median(rd)),
                     reproj_px=float(rec.compute_mean_reprojection_error()), num_points=int(rec.num_points3D()),
                     sim3_scale=float(sim.scale))
    json.dump(mj, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("REFINE_DONE")


def splits(body):
    """scene_split{k}/ for every scene/splits/test_split{k}.txt: symlinks to the shared images, masks and
    sparse model plus that split's test list (refine does this itself; the Itokawa scene of
    itokawa_prep.py / itokawa_refine.py needs it as a separate step)."""
    base = os.path.join(ROOT, body, "scene")
    for f in sorted(glob.glob(os.path.join(base, "splits", "test_split*.txt"))):
        k = os.path.basename(f)[len("test_split"):-4]
        d = os.path.join(ROOT, body, f"scene_split{k}")
        os.makedirs(d, exist_ok=True)
        for sub in ["images", "masks", "sparse"]:
            link = os.path.join(d, sub)
            if not os.path.lexists(link):
                os.symlink(os.path.join(base, sub), link)
        with open(os.path.join(d, "test_images.txt"), "w") as g:
            g.write(open(f).read())
        print(d, sum(1 for l in open(f) if l.strip()), "test images")


# ---------------------------------------------------------------- register (evaluation only)
def register(body):
    """Camera-frame -> reference-frame similarity from the SfM tie points (trimmed ICP).
    SPICE-anchored frames differ from the SPC/SfM reference frames by tens of metres and
    ~1% in scale; one transform per scene, shared by every method that uses these poses,
    keeps the comparison independent of each method's own mesh. Never seen by any method."""
    import trimesh
    from scipy.spatial import cKDTree
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))
    from eval_mesh import closest, raycast_scene, umeyama
    d = os.path.join(ROOT, body)
    P = np.loadtxt(os.path.join(d, "scene", "sparse", "0", "points3D.txt"), usecols=(1, 2, 3))
    ref = load_ref(d)
    dst = np.asarray(trimesh.sample.sample_surface(ref, 400000, seed=1)[0])
    tree, T = cKDTree(dst), np.eye(4)
    for gate in [0.06, 0.02, 0.006, 0.003]:
        for _ in range(40):
            cur = P @ T[:3, :3].T + T[:3, 3]
            dd, j = tree.query(cur, workers=-1)
            k = dd < gate
            T = umeyama(cur[k], dst[j[k]], True) @ T
    res, _ = closest(raycast_scene(ref), P @ T[:3, :3].T + T[:3, 3])
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    out = dict(matrix=T.tolist(), scale=s, translation_m=(T[:3, 3] * 1000).tolist(), n_points=len(P),
               sfm_residual_m=dict(median=float(np.median(res) * 1000), mean=float(res.mean() * 1000),
                                   p90=float(np.percentile(res, 90) * 1000)),
               note="evaluation-only registration of the SfM frame to the reference frame")
    json.dump(out, open(os.path.join(d, "scene", "frame_to_ref.json"), "w"), indent=1)
    print(f"{body}: frame->ref scale {s:.5f}, t {np.round(T[:3, 3] * 1000, 1)} m; SfM residual "
          f"median {out['sfm_residual_m']['median']:.2f} m, mean {out['sfm_residual_m']['mean']:.2f} m")


if __name__ == "__main__":
    {"geometry": geometry, "prep": prep, "refine": refine, "splits": splits,
     "register": register}[sys.argv[2]](sys.argv[1])
