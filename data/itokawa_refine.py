# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""SPICE-initialised triangulation + bundle adjustment for narrow-FOV imaging.

Free incremental SfM on the 60 AMICA frames (5.8 deg FOV, ~weak perspective)
converges to the Necker-reversed solution: relative rotations match SPICE only
after conjugation by diag(-1,-1,1) and the camera ring runs backwards. Starting
from SPICE poses puts BA in the correct basin; ~30 px pointing errors are then
removed by BA. The result is similarity-aligned to the SPICE camera centres
(ITOKAWA_FIXED, km). Overwrites scene/sparse/0 and the SfM fields of meta.json.
Step 3 of 3 for Itokawa (after data/itokawa_spice.py, data/itokawa_prep.py); its helpers are
also used by data/build_scene.py refine.
usage: python data/itokawa_refine.py
"""
import json
import os
import sys

import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itokawa_prep import OUT, pose, write_colmap_text  # noqa: E402

ROUNDS = [(40.0, 20.0), (8.0, 4.0), (4.0, 2.0)]  # (triangulation max reproj px, filter px)


def set_pose(rec, image, R_w2c, t):
    rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R_w2c), t)
    frame = rec.frame(image.frame_id)
    frame.rig_from_world = rig_from_world


def rel_rot_agreement(rec, meta, conj=np.eye(3)):
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    errs = []
    for i in range(0, len(ims), 5):
        for j in range(i + 1, len(ims), 5):
            Ri, Rj = pose(ims[i])[0], pose(ims[j])[0]
            Si, Sj = (np.array(meta[ims[k].name]["R_cam2body"]) for k in (i, j))
            Ra, Rb = Ri @ Rj.T, conj @ Si.T @ Sj @ conj
            errs.append(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
    return float(np.median(errs))


def main():
    db = os.path.join(OUT, "colmap", "database.db")
    rec = pycolmap.Reconstruction(os.path.join(OUT, "colmap", "0"))
    mj = json.load(open(os.path.join(OUT, "meta.json")))
    meta = mj["images"]
    print("free SfM: rel-rot agreement id %.3f deg, conj(-1,-1,1) %.3f deg" % (
        rel_rot_agreement(rec, meta), rel_rot_agreement(rec, meta, np.diag([-1., -1., 1.]))))

    for im in rec.images.values():
        Rc2b = np.array(meta[im.name]["R_cam2body"])
        C = np.array(meta[im.name]["cam_center_km"])
        R = Rc2b.T
        set_pose(rec, im, R, -R @ C)
    print("SPICE init: rel-rot agreement %.3f deg" % rel_rot_agreement(rec, meta))

    opts = pycolmap.IncrementalPipelineOptions()
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_focal_length = False
    ba.refine_principal_point = False
    ba.refine_extra_params = False
    for k, (tri_px, filt_px) in enumerate(ROUNDS):
        opts.triangulation.complete_max_reproj_error = tri_px
        opts.triangulation.merge_max_reproj_error = tri_px
        opts.triangulation.create_max_angle_error = 2.0 * tri_px / 10.0
        opts.mapper.filter_max_reproj_error = filt_px
        rec = pycolmap.triangulate_points(rec, db, os.path.join(OUT, "images"),
                                          os.path.join(OUT, "colmap", f"tri{k}"), clear_points=True,
                                          options=opts)
        pycolmap.bundle_adjustment(rec, ba)
        print(f"round {k}: points {rec.num_points3D()} reproj {rec.compute_mean_reprojection_error():.3f} px "
              f"rel-rot vs SPICE {rel_rot_agreement(rec, meta):.3f} deg")

    names = [im.name for im in rec.images.values()]
    tgt = np.array([meta[n]["cam_center_km"] for n in names])
    ro = pycolmap.RANSACOptions()
    ro.max_error = 0.1
    sim = pycolmap.align_reconstruction_to_locations(rec, names, tgt, 3, ro)
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
    cd, rd = np.array(cd), np.array(rd)
    print(f"Sim3 scale {sim.scale:.5f}; |C - C_spice| median {np.median(cd):.1f} m max {cd.max():.1f} m; "
          f"pointing correction median {np.median(rd):.3f} deg max {rd.max():.3f} deg")
    pts = np.array([p.xyz for p in rec.points3D.values()])
    r = np.linalg.norm(pts, axis=1)
    print(f"points radius km: median {np.median(r):.3f} p99 {np.percentile(r, 99):.3f}")

    write_colmap_text(rec, os.path.join(OUT, "sparse/0"), names)
    os.makedirs(os.path.join(OUT, "colmap", "refined"), exist_ok=True)
    rec.write(os.path.join(OUT, "colmap", "refined"))
    mj["sfm"] = dict(method="SPICE-initialised triangulation + 3 rounds BA, Sim3 to SPICE centres",
                     center_diff_m_median=float(np.median(cd)), pointing_corr_deg_median=float(np.median(rd)),
                     reproj_px=float(rec.compute_mean_reprojection_error()), num_points=int(rec.num_points3D()),
                     sim3_scale=float(sim.scale))
    json.dump(mj, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("REFINE_DONE")


if __name__ == "__main__":
    main()
