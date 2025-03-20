#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from PIL import Image
from pathlib import Path

from Style_3DGS.AdaIN.test import get_style_embeddings
from Style_3DGS.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from Style_3DGS.gaussian_renderer import render
import torchvision
from Style_3DGS.utils.general_utils import safe_state
from argparse import ArgumentParser
from Style_3DGS.arguments import ModelParams, PipelineParams, get_combined_args, get_combined_args_simple
from Style_3DGS.gaussian_renderer import GaussianModel
import torch.nn.functional as F

def create_gif(image_folder, gif_path, duration=0.1):
    image_paths = sorted(Path(image_folder).glob("*.png"))  # Sort to maintain sequence
    images = [Image.open(str(img_path)) for img_path in image_paths]
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0
    )

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    return render_path

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                style_image: Image.Image):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.precompute()

        style_f = get_style_embeddings(style_image).detach()
        style_f_pooled = F.adaptive_avg_pool2d(style_f, (1, 1))  # -> [1, 512, 1, 1]
        style_f_flat = style_f_pooled.reshape(style_f_pooled.size(0), -1)  # -> [1, 512]
        gaussians.style_f = style_f_flat

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            train_renders = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_gif_path = os.path.join(train_renders, "train_render.gif")
            create_gif(train_renders, train_gif_path)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

    return train_gif_path


def run_3dgs_rendering(style_image, model_path):
    """
    Run the rendering with minimal arguments.

    :param style_image: Path to the style image.
    :param model_path: Path to load trained 3DGS from
    """

    mocked_args = [
        f"--model_path={model_path}",
        "--skip_test"
    ]

    # Set up default parameter objects
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    global args
    args = parser.parse_args(mocked_args)
    args = get_combined_args_simple(args)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    style_path = Path(style_image)
    style_image = Image.open(str(style_path))
    
    gif_path = render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, style_image)
    return gif_path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument('--style_image', type=str, default="style_transfer/style/mondrian.jpg")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    style_path = Path(args.style_image)
    style_image = Image.open(str(style_path))
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, style_image)