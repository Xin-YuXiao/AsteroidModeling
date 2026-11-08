# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Evaluate an image-based asteroid reconstruction against a reference shape.

Group A reproduces the A&A protocols exactly:
  Asteroid-GS (Zhang+2025, A&A 704 A247, Eqs. 29-31): mean / RMSE / std of the
    unsigned distance from reconstructed-mesh points to the reference, in metres;
    volume, surface area, average edge length (A.E.L.); PSNR/SSIM/LPIPS on test views.
  Chen+2025 (A&A 696 A212, Table 2): Accuracy@tau = fraction of reconstructed
    vertices within tau metres of the reference, tau = 1..5 m.
  Both are one-sided (rec -> ref) and vertex-based, so holes are not penalised.
Group B adds what that protocol misses: ref -> rec completeness, symmetric
  Chamfer and F-score on area-uniform samples, normal angle error, watertightness,
  and polar / concave stratification. Foreground-masked PSNR is reported next to
  the full-frame PSNR because the black background dominates the latter.

Meshes are in the body-fixed frame in km (SPICE convention); outputs are metres.
Paper protocol (scripts/run_campaign.sh): --keep-largest --align similarity frame --samples 1000000,
--frame-tf <scene>/frame_to_ref.json; the tables report the 'frame' block.
usage: python evaluation/eval_mesh.py --rec <mesh.ply> --ref <ref_mesh_km.ply> --out <eval.json> [options]
"""
import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
import trimesh

TAUS_ACC = [1.0, 2.0, 3.0, 4.0, 5.0]      # m, Chen+2025
TAUS_F = [0.25, 0.5, 1.0, 2.0]            # m, F-score thresholds
FRAME_TF = None                           # set from --frame-tf


def components(mesh):
    """Connected-component label of every face, faces joined by shared edges (the grouping of
    trimesh's split(), which also builds one Trimesh per fragment and took >200 GB on GOF
    meshes full of floaters)."""
    return trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))


def raycast_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))))
    return scene


def closest(scene, query, chunk=500000):
    """Unsigned distance and closest-primitive id for each query point."""
    dist, prim = [], []
    for i in range(0, len(query), chunk):
        q = o3d.core.Tensor(query[i:i + chunk].astype(np.float32))
        r = scene.compute_closest_points(q)
        p = r["points"].numpy().astype(np.float64)
        dist.append(np.linalg.norm(p - query[i:i + chunk], axis=1))
        prim.append(r["primitive_ids"].numpy())
    return np.concatenate(dist), np.concatenate(prim)


def stats_m(d_km):
    d = d_km * 1000.0
    return dict(mean=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                std=float(d.std()), median=float(np.median(d)),
                p95=float(np.percentile(d, 95)), max=float(d.max()))


def mesh_props(mesh):
    """Watertightness, closedness, enclosed volume, area, mean unique-edge length (m) with vectorised
    numpy; trimesh's properties take minutes on multi-million-face meshes."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    key = e[:, 0] * (len(v) + 1) + e[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    eu = np.stack([uniq // (len(v) + 1), uniq % (len(v) + 1)], 1)
    tri = v[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    watertight = bool(len(cnt) and np.all(cnt == 2))
    # closed: no boundary edge. Screened Poisson meshes can carry a handful of non-manifold edges
    # (<= 5 of ~1e7 in the paper) that leave the enclosed volume well defined.
    boundary, nonmanifold = int((cnt == 1).sum()), int((cnt > 2).sum())
    closed = bool(len(cnt)) and boundary == 0
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    return dict(watertight=watertight, closed=closed, boundary_edges=boundary, nonmanifold_edges=nonmanifold,
                volume=abs(vol) if closed else float("nan"),
                area=float(0.5 * np.linalg.norm(cr, axis=1).sum()),
                ael_m=float(np.linalg.norm(v[eu[:, 0]] - v[eu[:, 1]], axis=1).mean() * 1000.0))


def ref_props(ref, path):
    cache = os.path.splitext(path)[0] + ".props.json"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return json.load(open(cache))
    p = mesh_props(ref)
    json.dump(p, open(cache, "w"))
    return p


def latitude_deg(pts):
    return np.degrees(np.arcsin(np.clip(pts[:, 2] / np.linalg.norm(pts, axis=1), -1, 1)))


def concave_mask(ref, pts, depth_km):
    """Points lying deeper than depth_km inside the reference convex hull
    (concave/convex stratification)."""
    hull = ref.convex_hull
    d, _ = closest(raycast_scene(hull), pts)
    return d > depth_km


def umeyama(src, dst, with_scale):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / xs.var(0).sum() if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, md - s * R @ ms
    return T


def register(rec, ref, mode, n=200_000, seed=3):
    """ICP of the reconstruction onto the reference: point-to-point Umeyama steps
    with a shrinking correspondence gate (scipy cKDTree; Open3D's KDTreeFlann
    segfaulted on our machines). mode: none | rigid | similarity."""
    if mode == "none":
        return rec, dict(mode="none")
    if mode == "frame":  # shared camera-frame -> reference registration (data/build_scene.py register)
        T = np.array(json.load(open(FRAME_TF))["matrix"])
        moved = rec.copy()
        moved.apply_transform(T)
        return moved, dict(mode="frame", source=FRAME_TF, scale=float(np.cbrt(np.linalg.det(T[:3, :3]))),
                           translation_m=(T[:3, 3] * 1000).tolist())
    from scipy.spatial import cKDTree
    src = np.asarray(trimesh.sample.sample_surface(rec, n, seed=seed)[0])
    src = src[np.isfinite(src).all(1)]
    dst = np.asarray(trimesh.sample.sample_surface(ref, n, seed=seed + 1)[0])
    tree = cKDTree(dst)
    T = np.eye(4)
    for max_d in [0.03, 0.01, 0.003]:  # km
        for _ in range(30):
            cur = src @ T[:3, :3].T + T[:3, 3]
            d, j = tree.query(cur, workers=-1)
            keep = np.isfinite(d) & (d < max_d)
            if keep.sum() < 100:
                break
            dT = umeyama(cur[keep], dst[j[keep]], mode == "similarity")
            T = dT @ T
            if np.abs(dT - np.eye(4)).max() < 1e-7:
                break
    moved = rec.copy()
    moved.apply_transform(T)
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    R = T[:3, :3] / s
    return moved, dict(mode=mode, scale=s, translation_m=(T[:3, 3] * 1000).tolist(),
                       rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                       matrix=T.tolist())


def eval_geometry(rec, ref, n_samples=1_000_000, seed=0, concave_depth_km=None, ref_p=None):
    ref_scene, rec_scene = raycast_scene(ref), raycast_scene(rec)
    rec_p = mesh_props(rec)
    ref_p = ref_p or mesh_props(ref)
    out = {}

    # --- Group A: A&A convention, reconstructed vertices -> reference
    v = np.asarray(rec.vertices, dtype=np.float64)
    d_v, _ = closest(ref_scene, v)
    out["A_vertex"] = stats_m(d_v)
    out["A_acc"] = {f"acc@{t:g}m": float((d_v * 1000 <= t).mean()) for t in TAUS_ACC}
    out["A_integral"] = dict(
        volume_km3=rec_p["volume"], area_km2=rec_p["area"], ael_m=rec_p["ael_m"],
        ref_volume_km3=ref_p["volume"], ref_area_km2=ref_p["area"], ref_ael_m=ref_p["ael_m"])
    a = out["A_integral"]
    a["volume_err_rel"] = (a["volume_km3"] - a["ref_volume_km3"]) / a["ref_volume_km3"]
    a["area_err_rel"] = (a["area_km2"] - a["ref_area_km2"]) / a["ref_area_km2"]

    # --- Group B: area-uniform samples, both directions
    p_rec, f_rec = trimesh.sample.sample_surface(rec, n_samples, seed=seed)
    p_ref, f_ref = trimesh.sample.sample_surface(ref, n_samples, seed=seed + 1)
    p_rec, p_ref = np.asarray(p_rec), np.asarray(p_ref)
    d_acc, prim_acc = closest(ref_scene, p_rec)     # accuracy (rec -> ref)
    d_comp, _ = closest(rec_scene, p_ref)           # completeness (ref -> rec)
    out["B_rec2ref"] = stats_m(d_acc)
    out["B_ref2rec"] = stats_m(d_comp)
    out["B_chamfer_m"] = 0.5 * (out["B_rec2ref"]["mean"] + out["B_ref2rec"]["mean"])
    fs = {}
    for t in TAUS_F:
        prec = float((d_acc * 1000 <= t).mean())
        rcl = float((d_comp * 1000 <= t).mean())
        fs[f"F@{t:g}m"] = dict(precision=prec, recall=rcl,
                               f=2 * prec * rcl / max(prec + rcl, 1e-12))
    out["B_fscore"] = fs
    n_rec = np.asarray(rec.face_normals)[f_rec]
    n_ref = np.asarray(ref.face_normals)[prim_acc]
    ang = np.degrees(np.arccos(np.clip(np.abs((n_rec * n_ref).sum(1)), 0, 1)))
    out["B_normal_angle_deg"] = dict(mean=float(ang.mean()), median=float(np.median(ang)))
    out["B_topology"] = dict(watertight=rec_p["watertight"],
                             components=int(len(np.unique(components(rec)))),
                             vertices=int(len(rec.vertices)), faces=int(len(rec.faces)))

    # --- stratification on reference samples (completeness) and their rec->ref
    lat = latitude_deg(p_ref)
    polar = np.abs(lat) > 60
    strata = {"polar": polar, "nonpolar": ~polar}
    if concave_depth_km is not None:
        cm = concave_mask(ref, p_ref, concave_depth_km)
        strata.update(concave=cm, convex=~cm)
    out["B_strata_ref2rec"] = {k: dict(frac=float(m.mean()), **stats_m(d_comp[m]))
                               for k, m in strata.items() if m.any()}
    return out


def load_gray(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    return im.astype(np.float64) / (65535.0 if im.dtype == np.uint16 else 255.0)


def eval_nvs(render_dir, gt_dir, mask_dir=None):
    """PSNR/SSIM/LPIPS between test renders and ground-truth images matched by
    file stem. Full-frame values follow Asteroid-GS (8-bit range); fg_psnr is
    computed over the union of the observed mask and the rendered foreground."""
    import torch
    import lpips
    from skimage.metrics import structural_similarity
    loss_fn = lpips.LPIPS(net="vgg", verbose=False)
    rows = []
    for gp in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(gp))[0]
        cand = glob.glob(os.path.join(render_dir, stem + ".*"))
        if not cand:
            continue
        gt, rd = load_gray(gp), load_gray(cand[0])
        mse = ((gt - rd) ** 2).mean()
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
        ssim = float(structural_similarity(gt, rd, data_range=1.0))
        def t(x):
            return torch.from_numpy(x).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1

        with torch.no_grad():
            lp = float(loss_fn(t(gt), t(rd)).item())
        row = dict(image=stem, psnr=psnr, ssim=ssim, lpips=lp)
        if mask_dir and os.path.exists(os.path.join(mask_dir, stem + ".png")):
            m = load_gray(os.path.join(mask_dir, stem + ".png")) > 0.5
            fg = m | (rd > 2.0 / 255.0)
            mse_fg = ((gt - rd)[fg] ** 2).mean()
            row["fg_psnr"] = float(10 * np.log10(1.0 / max(mse_fg, 1e-12)))
        rows.append(row)
    agg = {k: float(np.mean([r[k] for r in rows if k in r]))
           for k in ["psnr", "ssim", "lpips", "fg_psnr"] if any(k in r for r in rows)}
    return dict(n=len(rows), mean=agg, per_image=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="reconstructed mesh, body-fixed km")
    ap.add_argument("--ref", required=True, help="reference mesh, body-fixed km")
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", help="dir of test-view renders (png)")
    ap.add_argument("--gt-images", help="dir of test-view ground-truth images (png)")
    ap.add_argument("--masks", help="dir of foreground masks (png)")
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--concave-depth-km", type=float, default=None)
    ap.add_argument("--align", nargs="+", default=["none", "rigid", "similarity"],
                    help="registration modes to evaluate (none|frame|rigid|similarity); "
                         "each gets its own geometry block")
    ap.add_argument("--frame-tf", help="frame_to_ref.json for --align frame")
    ap.add_argument("--keep-largest", action="store_true",
                    help="drop all but the largest connected component before evaluation")
    a = ap.parse_args()
    global FRAME_TF
    FRAME_TF = a.frame_tf

    rec = trimesh.load(a.rec, force="mesh", process=False)
    ref = trimesh.load(a.ref, force="mesh", process=False)
    bad = ~np.isfinite(rec.vertices).all(1)
    if bad.any():  # drop faces touching non-finite vertices (seen in some TSDF outputs)
        rec.update_faces(~bad[rec.faces].any(1))
        rec.remove_unreferenced_vertices()
    if a.keep_largest:
        lab = components(rec)
        rec.update_faces(lab == np.bincount(lab).argmax())
        rec.remove_unreferenced_vertices()
    res = dict(rec=a.rec, ref=a.ref, geometry_by_align={})
    rp = ref_props(ref, a.ref)
    for mode in a.align:
        moved, tf = register(rec, ref, mode)
        g = eval_geometry(moved, ref, a.samples, concave_depth_km=a.concave_depth_km, ref_p=rp)
        g["registration"] = tf
        res["geometry_by_align"][mode] = g
    res["geometry"] = res["geometry_by_align"][a.align[-1]]
    if a.renders and a.gt_images:
        res["nvs"] = eval_nvs(a.renders, a.gt_images, a.masks)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    for mode, g in res["geometry_by_align"].items():
        print(f"--- align={mode} {({k: v for k, v in g['registration'].items() if k != 'matrix'})}")
        print(f"[A] mean {g['A_vertex']['mean']:.3f} rmse {g['A_vertex']['rmse']:.3f} "
              f"std {g['A_vertex']['std']:.3f} m | acc@1m {g['A_acc']['acc@1m']:.3f} | "
              f"vol err {100 * g['A_integral']['volume_err_rel']:+.2f}%")
        print(f"[B] chamfer {g['B_chamfer_m']:.3f} m | ref->rec {g['B_ref2rec']['mean']:.3f} m | "
              f"F@1m {g['B_fscore']['F@1m']['f']:.3f} | normal {g['B_normal_angle_deg']['mean']:.2f} deg | "
              f"wt {g['B_topology']['watertight']} comps {g['B_topology']['components']}")
    if "nvs" in res:
        print(f"[NVS] {res['nvs']['mean']}")


if __name__ == "__main__":
    main()
