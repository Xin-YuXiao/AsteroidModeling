# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""Reference-free global correction of any mesh against the scene's SfM tie points, applied
to every method alike (fairness control for the main table).

Two global errors show up in the splatting reconstructions: a rigid shift of about one pixel
(Asteroid-GS and PGSR sit ~1 px along the spin axis, probably the 3DGS/2DGS pixel convention
that PCGS corrects with --pixfix) and an offset along the normals (rendered depth in front of
or behind the surface). Both are estimated from the COLMAP tie points (reprojection error
<= max_err px), which owe nothing to any splatting code. The signed point-to-plane residuals
of the tie points to the closest mesh points are fitted with r_i = n_i . t + delta (|r| < gate,
Huber weights, a few iterations); the mesh is then offset by delta along its vertex normals
and moved by t. No reference model is used.

usage: python evaluation/align_to_sfm.py <scene_dir> <mesh.ply> <out.ply>     (scene units: km)
"""
import json
import sys

import numpy as np
import open3d as o3d
import trimesh

scene, src, dst = sys.argv[1:4]
max_err, gate, huber, iters = 1.0, 0.005, 0.001, 20      # px, 5 m, 1 m, iterations
A = np.loadtxt(f"{scene}/sparse/0/points3D.txt", usecols=(1, 2, 3, 7), comments="#")
X = A[A[:, 3] <= max_err, :3]
m = trimesh.load(src, process=False)
V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
s = o3d.t.geometry.RaycastingScene()
s.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
fn = np.asarray(m.face_normals)
t, delta = np.zeros(3), 0.0
for it in range(iters):
    Y = X - t                                             # tie points in the uncorrected mesh's frame
    r_ = s.compute_closest_points(o3d.core.Tensor(Y.astype(np.float32)))
    p, n = r_["points"].numpy().astype(np.float64), fn[r_["primitive_ids"].numpy()]
    r = ((Y - p) * n).sum(1) - delta                      # residual to the corrected surface
    k = np.abs(r) < gate
    w = np.minimum(1.0, huber / np.maximum(np.abs(r[k]), 1e-9))
    J = np.concatenate([n[k], np.ones((k.sum(), 1))], 1) * np.sqrt(w)[:, None]
    step = np.linalg.lstsq(J, r[k] * np.sqrt(w), rcond=None)[0]
    t, delta = t + step[:3], delta + step[3]
    if np.abs(step).max() < 1e-6:                         # 1 mm
        break
out = trimesh.Trimesh(V + delta * np.asarray(m.vertex_normals) + t, F, process=False)
out.export(dst)
print("ALIGN", json.dumps(dict(points=int(k.sum()), iterations=it + 1, t_m=[round(1000 * v, 3) for v in t],
                               delta_m=round(1000 * delta, 3), median_abs_res_m=round(1000 * float(np.median(np.abs(r[k]))), 3))))
