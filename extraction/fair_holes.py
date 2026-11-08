# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Biharmonic fairing of the unobserved parts of a watertight PCGS mesh.

Vertices of mesh_mvpoisson farther than `gap` from the consistency-filtered TSDF
mesh (mesh_mvtsdf) have no multi-view support: screened Poisson interpolates
them, and on a development Itokawa model these 1.2 % of vertices sat ~8 m inside the body
and carried 75 % of the squared error. Here they (plus `ring`
rings around them) are re-solved as the biharmonic surface (uniform bi-Laplacian,
L^2 x = 0) bounded by the supported vertices, i.e. the smoothest continuation of
the observed surface. Topology is unchanged, so the mesh stays watertight. Uses no
reference model.

usage: python extraction/fair_holes.py <model_dir> [--gap 0.002] [--ring 2] [--tag _fair]
Writes <run_dir>/mesh_mvpoisson<tag>.ply.
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh


def dist_to(mesh, pts):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                                                o3d.core.Tensor(np.asarray(mesh.faces, np.uint32))))
    cp = s.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))["points"].numpy()
    return np.linalg.norm(pts - cp, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gap", type=float, default=0.002, help="km")
    ap.add_argument("--ring", type=int, default=2)
    ap.add_argument("--tag", default="_fair")
    a = ap.parse_args()
    pm = trimesh.load(os.path.join(a.run, "mesh_mvpoisson.ply"), process=False)
    tm = trimesh.load(os.path.join(a.run, "mesh_mvtsdf.ply"), process=False)
    V, Fc = np.asarray(pm.vertices, np.float64), np.asarray(pm.faces)
    n = len(V)
    e = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    free = dist_to(tm, V) > a.gap
    n0 = int(free.sum())
    for _ in range(a.ring):
        free = free | (A @ free.astype(np.float64) > 0)
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    B = (L @ L).tocsr()
    fi, ci = np.flatnonzero(free), np.flatnonzero(~free)
    Bf = B[fi]
    solve = spla.factorized(Bf[:, fi].tocsc())
    rhs = -(Bf[:, ci] @ V[ci])
    Vn = V.copy()
    for k in range(3):
        Vn[fi, k] = solve(rhs[:, k])
    move = np.linalg.norm(Vn - V, axis=1) * 1000
    out = trimesh.Trimesh(Vn, Fc, process=False)
    out.export(os.path.join(a.run, f"mesh_mvpoisson{a.tag}.ply"))
    info = dict(unsupported=n0, solved=len(fi), frac=len(fi) / n, move_mean_m=float(move[fi].mean()),
                move_p95_m=float(np.percentile(move[fi], 95)), watertight=bool(out.is_watertight))
    json.dump(info, open(os.path.join(a.run, f"fair{a.tag}.json"), "w"), indent=1)
    print("FAIR", json.dumps(info))


if __name__ == "__main__":
    main()
