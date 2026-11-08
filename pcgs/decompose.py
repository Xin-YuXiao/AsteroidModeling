# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Photometric decomposition of PCGS renders for the figures: for chosen views, the observed image,
the predicted image (albedo x lunar-Lambert shading under the view's SPICE Sun, no cast shadows,
as for novel views), the albedo map, the shading map Phi, the surfel normals (camera frame) and the
opacity. Saved as <out>/<view>.npz (float16 arrays, HxW or 3xHxW).
usage: python pcgs/decompose.py -s <scene_split> -m <model_dir> --sun_shading --views <name> [<name> ...]
       --out <dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from argparse import ArgumentParser

import numpy as np
import torch

import pcgs.extras as pcgs
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sun_shading", action="store_true")
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if dataset.pixfix:
        pcgs.apply_pixfix(scene, dataset.source_path)
    bg = torch.zeros(3, device="cuda")
    aux = pcgs.SceneAux(dataset.source_path)
    phot = pcgs.Photometry.from_checkpoint(os.path.join(args.model_path, "phot.pt"))[0].cuda()
    os.makedirs(args.out, exist_ok=True)
    cams = {c.image_name: c for c in scene.getTrainCameras() + scene.getTestCameras()}
    with torch.no_grad():
        for v in args.views:
            v = os.path.splitext(v)[0]
            cam = cams[v]
            pkg = render(cam, gaussians, pipe, bg)
            pred, phi = pcgs.shade(pkg, cam, aux, phot, None)
            a = pkg["rend_alpha"].clamp(min=1e-3)
            n = pkg["rend_normal"]                                   # world frame
            Rw2c = cam.world_view_transform[:3, :3].T                 # 2DGS stores the transpose
            n_cam = (Rw2c @ n.reshape(3, -1)).reshape(n.shape)
            np.savez_compressed(os.path.join(args.out, v + ".npz"),
                                gt=cam.original_image[0].cpu().numpy().astype(np.float16),
                                pred=pred[0].cpu().numpy().astype(np.float16),
                                albedo=(pkg["render"][0] / a[0]).cpu().numpy().astype(np.float16),
                                phi=phi[0].cpu().numpy().astype(np.float16),
                                normal=n_cam.cpu().numpy().astype(np.float16),
                                alpha=pkg["rend_alpha"][0].cpu().numpy().astype(np.float16),
                                test=np.array(v in {c.image_name for c in scene.getTestCameras()}))
            print("DECOMPOSE", v, "L =", phot.summary())
