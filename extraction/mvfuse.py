# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Multi-view-consistent mesh extraction for a trained model (2DGS / PCGS).

usage: python extraction/mvfuse.py -s <scene_split> -m <model_dir> [--min_agree 3] [--tol 0.0015]
Writes <model>/mesh_mvpoisson.ply (watertight) and <model>/mesh_mvtsdf.ply, keeping only
median depths confirmed by >= min_agree neighbouring training views (pcgs.extras.mvfuse_meshes).
env OBJECT_EXTENT: object extent in km (points beyond 3 extents from the centre are dropped).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from argparse import ArgumentParser

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
    parser.add_argument("--min_agree", default=3, type=int)
    parser.add_argument("--tol", default=0.0015, type=float)
    parser.add_argument("--poisson_depth", default=11, type=int)
    parser.add_argument("--tag", default="")
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    pipe.depth_ratio = 1.0  # median depth for extraction (tail analysis: expected depth is 2-3x worse)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if dataset.pixfix:
        pcgs.apply_pixfix(scene, dataset.source_path)
    aux = pcgs.SceneAux(dataset.source_path)
    max_r = 3.0 * float(os.environ.get("OBJECT_EXTENT", "0.33"))
    pm, tm, info = pcgs.mvfuse_meshes(scene.getTrainCameras(), gaussians, pipe, torch.zeros(3, device="cuda"),
                                      render, aux, min_agree=args.min_agree, tol=args.tol,
                                      poisson_depth=args.poisson_depth, max_radius=max_r)
    pm.export(os.path.join(args.model_path, f"mesh_mvpoisson{args.tag}.ply"))
    tm.export(os.path.join(args.model_path, f"mesh_mvtsdf{args.tag}.ply"))
    info.update(poisson_faces=len(pm.faces), poisson_watertight=bool(pm.is_watertight), tsdf_faces=len(tm.faces))
    json.dump(info, open(os.path.join(args.model_path, f"mvfuse{args.tag}.json"), "w"), indent=1)
    print("MVFUSE", info)
