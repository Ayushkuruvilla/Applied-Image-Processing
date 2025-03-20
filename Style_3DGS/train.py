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

import sys
import os
from pathlib import Path
import torch
from random import randint

from PIL import Image
from torchvision.utils import make_grid

from Style_3DGS.AdaIN import adain_inference, get_style_embeddings
from Style_3DGS.utils.loss_utils import l1_loss, ssim
from Style_3DGS.gaussian_renderer import render, network_gui
from Style_3DGS.scene import Scene, GaussianModel
from Style_3DGS.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
from Style_3DGS.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from Style_3DGS.arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    comp,
    store_npz,
    freeze_iters=7000,
    style_img_path=None,
    use_depth=False,
    w_style=1e2,
    img_size=512,
    depth_offset=0.5,
    depth_prominence=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stylizing for {style_img_path}, using device: {device}")

    style_path = Path(style_img_path)
    style_image = Image.open(str(style_path))

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if dataset.style_dim:
        style_f = get_style_embeddings(style_image).detach()
        style_f_pooled = F.adaptive_avg_pool2d(style_f, (1, 1))  # -> [1, 512, 1, 1]
        style_f = style_f_pooled.view(1, 512)  # -> [1, 512]
        # store for later use
        gaussians.style_f = style_f.float()

    # Cache stylized views
    view_stylized_cache = {}
    for cam in scene.getTrainCameras():
        gt_image_np = (cam.original_image.squeeze(0).cpu().numpy() * 255).astype(
            "uint8"
        )
        gt_image = Image.fromarray(
            gt_image_np.transpose(1, 2, 0)
        )  # [C, H, W] -> [H, W, C]

        # Generate a binary mask from the ground truth image
        mask = gt_image_np > 0

        output_path = Path(dataset.model_path) / "stylized"
        # output_path = '../output/stylized'
        image_guide_path = adain_inference(
            content_img=gt_image,
            style_img=style_image,
            content_size=img_size,
            style_size=img_size,
            content_mask=mask,
            output=output_path,
            file_name=cam.image_name,
            use_depth=use_depth,
            depth_offset=depth_offset,
            depth_prominence=depth_prominence,
        )

        view_stylized_cache[cam.image_name] = image_guide_path
    print("\nFinished computing stylized views")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if iteration <= opt.rvq_iter:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                itr=iteration,
                rvq_iter=False,
            )
        else:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background, itr=iteration, rvq_iter=True
            )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = 0.0
        # loss_style = 0.0
        Ll1 = 0.0
        if iteration < freeze_iters:
            Ll1 = l1_loss(image, gt_image)
            loss = (
                (1.0 - opt.lambda_dssim) * Ll1
                + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                + opt.lambda_mask * torch.mean((torch.sigmoid(gaussians._mask)))
            )

        # Compute stylization guide loss
        if iteration >= freeze_iters:
            image_guide_path = view_stylized_cache[viewpoint_cam.image_name]
            image_guide = Image.open(str(image_guide_path)).convert("RGB")
            to_tensor = T.ToTensor()
            image_guide_tensor = to_tensor(image_guide).unsqueeze(0).to(image.device)
            # Resize `image_guide` to match the size of `image`
            image_guide_resized = F.interpolate(
                image_guide_tensor,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            Ll1 = l1_loss(image, image_guide_resized)
            loss = Ll1

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "L1 Loss": f"{Ll1:.{7}f}; Loss: {ema_loss_for_log:.{7}f}"
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration == opt.iterations:
                storage = gaussians.final_prune(compress=comp)
                with open(os.path.join(args.model_path, "storage"), "w") as c:
                    c.write(storage)
                gaussians.precompute()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, compress=comp, store=store_npz)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            else:
                if iteration % opt.mask_prune_iter == 0:
                    gaussians.mask_prune()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer_net.step()
                gaussians.optimizer_net.zero_grad(set_to_none=True)
                gaussians.scheduler_net.step()
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


def run_3dgs_training(
    source_path,
    style_image,
    output_folder,
    use_depth=False,
    iterations=15000,
    freeze_iters=7000,
    depth_offset=0.5,
    depth_prominence=20,
):
    """
    Run the training with minimal arguments.

    :param source_path: Path to the source images.
    :param style_image: Path to the style image.
    :param output_folder: Path to save output files.
    :param use_depth: Whether to use depth during training (default: False).
    :param iterations: Number of iterations (default: 15000).
    """

    mocked_args = [
        f"--source_path={source_path}",
        f"--model_path={output_folder}",
        f"--iterations={iterations}",
    ]

    # Set up default parameter objects
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--comp", action="store_true")
    parser.add_argument("--store_npz", action="store_true")

    global args
    args = parser.parse_args(mocked_args)
    args.save_iterations.append(args.iterations)

    print(
        f"Running training with source: {args.source_path}, style: {style_image}, "
        f"output: {args.model_path}, use_depth: {use_depth}, iterations: {args.iterations}"
    )

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print(args.source_path)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.comp,
        args.store_npz,
        freeze_iters=freeze_iters,
        style_img_path=style_image,
        use_depth=use_depth,
        w_style=0.5,
        img_size=512,
        depth_offset=depth_offset,
        depth_prominence=depth_prominence,
    )

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--comp", action="store_true")
    parser.add_argument("--store_npz", action="store_true")

    parser.add_argument(
        "--style_image", type=str, default="style_transfer/style/mondrian.jpg"
    )
    parser.add_argument("--freeze_iters", type=int, default=7000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print(args.source_path)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.comp,
        args.store_npz,
        freeze_iters=args.freeze_iters,
        style_img_path=args.style_image,
        w_style=0.5,
        img_size=512,
    )

    # All done
    print("\nTraining complete.")
