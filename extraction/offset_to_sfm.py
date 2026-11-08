# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Global surface-offset correction of a PCGS mesh against its own SfM tie points.

The surfel median depth stops slightly in front of the surface: on development models the
scene's tie points sat a median 0.37 m (Ryugu, 2.2 m/px) and 0.10 m (Itokawa, 0.8 m/px)
inside the mesh, ~0.15 GSD on both. A per-point depth loss removes it but roughens
the surface (normal error +3-4 deg), so here one scalar is estimated instead:
the median signed distance of the tie points (reprojection error <= max_err px,
|d| < gate) to the mesh, applied as a shift along the area-weighted vertex
normals. Uses no reference model.

usage: python extraction/offset_to_sfm.py <scene_dir> <mesh.ply> <out.ply>     (scene units: km)
"""
import json
import sys

import numpy as np
import open3d as o3d
import trimesh

scene, src, dst = sys.argv[1:4]
max_err, gate = 1.0, 0.005
A = np.loadtxt(f"{scene}/sparse/0/points3D.txt", usecols=(1, 2, 3, 7), comments="#")
X = A[A[:, 3] <= max_err, :3]
m = trimesh.load(src, process=False)
s = o3d.t.geometry.RaycastingScene()
s.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
                                            o3d.core.Tensor(np.asarray(m.faces, np.uint32))))
r = s.compute_closest_points(o3d.core.Tensor(X.astype(np.float32)))
sd = ((X - r["points"].numpy()) * m.face_normals[r["primitive_ids"].numpy()]).sum(1)
delta = float(np.median(sd[np.abs(sd) < gate]))
out = trimesh.Trimesh(np.asarray(m.vertices) + delta * m.vertex_normals, m.faces, process=False)
out.export(dst)
print("OFFSET", json.dumps(dict(points=int((np.abs(sd) < gate).sum()), delta_m=delta * 1000)))
