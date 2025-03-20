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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from Style_3DGS.scene.gaussian_model import GaussianModel
from Style_3DGS.utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1, rvq_iter=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    cov3D_precomp = None

    if itr == -1:
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity

        xyz = pc.contract_to_unisphere(means3D.clone().detach(),
                                       torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        if pc.style_f is not None:
            style_embed = pc.style_fc(pc.style_f)
            norms = style_embed.norm(dim=1, keepdim=True)
            # we have to normalize to not overpower the color module
            style_embed = style_embed / norms
            style_embed = style_embed.repeat(xyz.shape[0], 1)
            recolor_out = pc.recolor(xyz)

            mlp_input = torch.cat([recolor_out, style_embed], dim=1)
        else:
            mlp_input = pc.recolor(xyz)
        shs = pc.mlp_head(mlp_input).unsqueeze(1)
        
    else:
        mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
        if rvq_iter:
            scales = pc.vq_scale(pc.get_scaling.unsqueeze(0))[0]
            rotations = pc.vq_rot(pc.get_rotation.unsqueeze(0))[0]
            scales = scales.squeeze()*mask
            rotations = rotations.squeeze()
            opacity = pc.get_opacity*mask

        else:
            scales = pc.get_scaling*mask
            rotations = pc.get_rotation
            opacity = pc.get_opacity*mask
            
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))

        if pc.style_f is not None:
            style_embed = pc.style_fc(pc.style_f)
            norms = style_embed.norm(dim=1, keepdim=True)
            # we have to normalize to not overpower the color module
            style_embed = style_embed / norms
            style_embed = style_embed.repeat(xyz.shape[0], 1)
            recolor_out = pc.recolor(xyz)

            mlp_input = torch.cat([recolor_out, style_embed], dim=1)
        else:
            mlp_input = pc.recolor(xyz)
        shs = pc.mlp_head(mlp_input).unsqueeze(1)

    # [n, 1, 48] -> [n, 16, 3]
    shs = shs.reshape(-1, 16, 3)

    # Save the current spherical harmonics as `f_dc` and `f_rest`
    pc._features_dc = shs[:, :1, :]  # The first basis (degree 0) as the direct component (R, G, B)
    pc._features_rest = shs[:, 1:, :]  # The remaining SH coefficients

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D.float(),
        means2D = means2D,
        shs = shs.float(),
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }