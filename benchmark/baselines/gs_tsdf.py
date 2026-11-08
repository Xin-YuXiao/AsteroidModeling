"""Mesh extraction by TSDF fusion of rendered depth, shared by the baselines that have no
extraction of their own at our scale (3DGS, Mip-Splatting) or a DTU-scale one (RaDe-GS).
Every method gets the same fusion as the 2DGS and PGSR rows of the main table: training views,
depth masked by the body mask, 1 m voxels, 4 m truncation, depth cut at the farthest camera
range + 2 object extents, largest connected cluster. Depth is rendered with each method's own
renderer and pixel convention:
  3dgs  graphdeco 3DGS renderer, "depth" = inverse depth
  gof   GOF renderer (used for Mip-Splatting models, which GOF can load): channels 6/7 = depth/alpha
  rade  RaDe-GS renderer, expected or median depth as its own extraction (depth_ratio)
Run from inside the method's repository with its interpreter (the repo's modules are imported):
  cd repos/<repo> && $PY .../gs_tsdf.py --method <m> -s <scene_split> -m <model> --depth_trunc <km> --out <mesh.ply>
"""
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch

sys.path.insert(0, os.getcwd())
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402


def depth_of(method, view, gaussians, pipe, bg, dataset):
    if method == "3dgs":
        inv = render(view, gaussians, pipe, bg)["depth"][0]
        return torch.where(inv > 1e-6, 1.0 / inv.clamp(min=1e-6), torch.zeros_like(inv))
    if method == "gof":
        r = render(view, gaussians, pipe, bg, kernel_size=dataset.kernel_size)["render"]
        return torch.where(r[7] > 0.5, r[6], torch.zeros_like(r[6]))
    if method == "rade":   # as RaDe-GS's own mesh_extract.py: expected depth unless depth_ratio >= 0.5
        name = "expected_depth" if dataset.depth_ratio < 0.5 else "median_depth"
        return render(view, gaussians, pipe, bg, dataset.kernel_size)[name][0]
    raise ValueError(method)


def intrinsics(view):
    W, H = view.image_width, view.image_height
    if hasattr(view, "Fx") and hasattr(view, "Cx"):          # RaDe-GS cameras carry their own
        return np.array([[view.Fx, 0, view.Cx], [0, view.Fy, view.Cy], [0, 0, 1]], np.float64)
    fx = W / (2 * np.tan(view.FoVx / 2))
    fy = H / (2 * np.tan(view.FoVy / 2))
    return np.array([[fx, 0, (W - 1) / 2], [0, fy, (H - 1) / 2], [0, 0, 1]], np.float64)   # 3DGS ndc2pix


def main():
    ap = ArgumentParser()
    model, pipeline = ModelParams(ap, sentinel=True), PipelineParams(ap)
    ap.add_argument("--method", required=True, choices=("3dgs", "gof", "rade"))
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--depth_trunc", type=float, required=True, help="km")
    ap.add_argument("--voxel", type=float, default=0.001)
    ap.add_argument("--sdf_trunc", type=float, default=0.004)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = get_combined_args(ap)
    dataset, pipe = model.extract(a), pipeline.extract(a)
    masks = os.path.join(dataset.source_path, "masks")
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=a.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda")
        dev = o3d.core.Device("CUDA:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight"), attr_dtypes=(o3c.float32, o3c.float32),
                                            attr_channels=((1), (1)), voxel_size=a.voxel, block_resolution=16,
                                            block_count=100000, device=dev)
        mult = a.sdf_trunc / a.voxel
        for view in scene.getTrainCameras():
            d = depth_of(a.method, view, gaussians, pipe, bg, dataset).float()
            mp = os.path.join(masks, view.image_name + ".png")
            if os.path.exists(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                d[~torch.from_numpy(m).cuda()] = 0
            d[d > a.depth_trunc] = 0
            depth = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d.cpu().numpy()))).to(dev)
            K = o3c.Tensor(intrinsics(view))
            E = o3c.Tensor(view.world_view_transform.T.cpu().numpy().astype(np.float64))
            blocks = vbg.compute_unique_block_coordinates(depth, K, E, 1.0, a.depth_trunc, mult)
            vbg.integrate(blocks, depth, K, E, 1.0, a.depth_trunc, mult)
        mesh = vbg.extract_triangle_mesh().to_legacy()
    tri, n, _ = mesh.cluster_connected_triangles()
    tri, n = np.asarray(tri), np.asarray(n)
    if len(n):
        mesh.remove_triangles_by_mask(tri != n.argmax())
        mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(a.out, mesh)
    print("TSDF_OK", a.method, a.out, len(mesh.triangles), "triangles")


if __name__ == "__main__":
    main()
