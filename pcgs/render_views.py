# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
"""Render the held-out views of a trained PCGS model.

If the model was trained with --sun_shading, each held-out view is rendered as the albedo map shaded by the
lunar-Lambert function under that view's SPICE Sun direction, with the learned parameter L (phot.pt) and no
per-image fitting. Cast shadows are not rendered, because their gate needs the observed image. Renders and
ground truth are written to test/ours_<it>/{renders,gt}/00000.png... in sorted test-name order.
usage: python pcgs/render_views.py -s <scene_split> -m <model_dir> --sun_shading
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from argparse import ArgumentParser

import torch
import torchvision

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
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if dataset.pixfix:
        pcgs.apply_pixfix(scene, dataset.source_path)
    bg = torch.zeros(3, device="cuda")
    it = scene.loaded_iter
    aux = pcgs.SceneAux(dataset.source_path)
    phot, cfg = pcgs.Photometry.from_checkpoint(os.path.join(args.model_path, "phot.pt"))
    phot = phot.cuda()
    shadow = cfg.get("shadow")
    # --shadow_gate training applies cast shadows only where the training image is darker than the
    # unshadowed prediction; a test view has no image to gate with, and the ungated Sun-view depth
    # test (sub-GSD depth noise) speckles the render with false shadows, so novel views are shaded
    # without cast shadows.
    if cfg.get("shadow_gate", 0) > 0:
        shadow = None
    min_phase = math.radians(cfg.get("shadow_min_phase", 0.0))
    extent = float(os.environ.get("OBJECT_EXTENT", "0.33"))
    print(f"sun_shading={args.sun_shading} phot={phot.summary()} shadows={shadow is not None}")

    out = os.path.join(args.model_path, "test", f"ours_{it}")
    os.makedirs(os.path.join(out, "renders"), exist_ok=True)
    os.makedirs(os.path.join(out, "gt"), exist_ok=True)
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    with torch.no_grad():
        for i, cam in enumerate(test):
            pkg = render(cam, gaussians, pipe, bg)
            vis = pcgs.cast_shadow_map(pkg, cam, gaussians, pipe, bg, render, aux, extent, **shadow) \
                if shadow and float(aux.alpha[cam.image_name]) >= min_phase else None
            img = pcgs.shade(pkg, cam, aux, phot, vis)[0] if args.sun_shading else pkg["render"]
            torchvision.utils.save_image(img.clamp(0, 1), os.path.join(out, "renders", f"{i:05d}.png"))
            torchvision.utils.save_image(cam.original_image[0:3], os.path.join(out, "gt", f"{i:05d}.png"))

