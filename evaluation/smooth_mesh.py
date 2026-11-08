# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Taubin smoothing of a reconstructed mesh (volume preserving; removes the sub-GSD vertex noise of
the depth-fusion + Poisson mesh without shrinking it).
Paper protocol: 20 iterations, applied to every method after evaluation/align_to_sfm.py.
usage: python evaluation/smooth_mesh.py <in.ply> <out.ply> [iterations=10] [lambda=0.5] [mu=-0.53]
"""
import sys

import open3d as o3d

src, dst = sys.argv[1], sys.argv[2]
it = int(sys.argv[3]) if len(sys.argv) > 3 else 10
lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
mu = float(sys.argv[5]) if len(sys.argv) > 5 else -0.53
m = o3d.io.read_triangle_mesh(src)
m = m.filter_smooth_taubin(number_of_iterations=it, lambda_filter=lam, mu=mu)
m.compute_vertex_normals()
o3d.io.write_triangle_mesh(dst, m)
print(f"SMOOTH {src} -> {dst}: {len(m.triangles)} faces, taubin {it} x ({lam}, {mu})")
