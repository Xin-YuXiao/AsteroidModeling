"""Distance distributions of one reconstruction for the result figures, with the main-table
protocol (evaluation/eval_mesh.py: non-finite faces dropped, largest component kept, frame registration,
area-uniform samples with the same seeds):

  acc     |d| of reconstruction samples to the reference (accuracy), 1 cm bins, 0-10 m
  comp    |d| of reference samples to the reconstruction (completeness), 1 cm bins, 0-10 m
  signed  signed d of reconstruction samples (+ outside the reference), 2.5 cm bins, -5..5 m
  normal  angle between reconstruction and closest-reference face normals, 0.5 deg bins
  lat     mean / median completeness distance per 5 deg latitude band of the reference samples
Counts outside a histogram's range are kept as under/overflow.
usage: python dist_profiles.py <rec.ply> <ref.ply> <frame_to_ref.json> <out.json> [samples=1000000]
"""
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import img_eval  # noqa: E402


def hist(x, lo, hi, step):
    edges = np.arange(lo, hi + step / 2, step)
    h, _ = np.histogram(x, edges)
    return dict(lo=lo, hi=hi, step=step, counts=h.tolist(), under=int((x < lo).sum()), over=int((x >= hi).sum()))


def main():
    rec_p, ref_p, tf, out = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 1_000_000
    img_eval.FRAME_TF = tf
    rec = trimesh.load(rec_p, force="mesh", process=False)
    ref = trimesh.load(ref_p, force="mesh", process=False)
    bad = ~np.isfinite(rec.vertices).all(1)
    if bad.any():
        rec.update_faces(~bad[rec.faces].any(1))
        rec.remove_unreferenced_vertices()
    lab = img_eval.components(rec)
    rec.update_faces(lab == np.bincount(lab).argmax())
    rec.remove_unreferenced_vertices()
    rec, _ = img_eval.register(rec, ref, "frame")

    ref_scene, rec_scene = img_eval.raycast_scene(ref), img_eval.raycast_scene(rec)
    p_rec, f_rec = trimesh.sample.sample_surface(rec, n, seed=0)
    p_ref, _ = trimesh.sample.sample_surface(ref, n, seed=1)
    p_rec, p_ref = np.asarray(p_rec), np.asarray(p_ref)
    d_acc, prim = img_eval.closest(ref_scene, p_rec)
    d_comp, _ = img_eval.closest(rec_scene, p_ref)
    sd = []
    for i in range(0, n, 500000):                  # sign from the closed reference (ray parity)
        q = o3d.core.Tensor(p_rec[i:i + 500000].astype(np.float32))
        sd.append(ref_scene.compute_signed_distance(q).numpy())
    sign = np.sign(np.concatenate(sd))
    n_rec = np.asarray(rec.face_normals)[f_rec]
    n_ref = np.asarray(ref.face_normals)[prim]
    ang = np.degrees(np.arccos(np.clip(np.abs((n_rec * n_ref).sum(1)), 0, 1)))
    lat = img_eval.latitude_deg(p_ref)
    bands = np.arange(-90, 91, 5)
    li = np.clip(np.digitize(lat, bands) - 1, 0, len(bands) - 2)
    dc = d_comp * 1000
    prof = dict(edges=bands.tolist(),
                mean=[float(dc[li == k].mean()) if (li == k).any() else None for k in range(len(bands) - 1)],
                median=[float(np.median(dc[li == k])) if (li == k).any() else None for k in range(len(bands) - 1)],
                frac=[float((li == k).mean()) for k in range(len(bands) - 1)])
    res = dict(rec=rec_p, samples=n, acc=hist(d_acc * 1000, 0, 10, 0.01), comp=hist(dc, 0, 10, 0.01),
               signed=hist(sign * d_acc * 1000, -5, 5, 0.025), normal=hist(ang, 0, 90, 0.5), lat=prof,
               summary=dict(acc_mean=float(d_acc.mean() * 1000), comp_mean=float(dc.mean()),
                            signed_median=float(np.median(sign * d_acc * 1000)), normal_mean=float(ang.mean())))
    json.dump(res, open(out, "w"))
    print("PROFILE", out, {k: round(v, 3) for k, v in res["summary"].items()})


if __name__ == "__main__":
    main()
