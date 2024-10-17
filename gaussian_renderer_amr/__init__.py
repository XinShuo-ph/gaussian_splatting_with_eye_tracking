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

import numpy as np
import torchvision

import torch
import math
from diff_gaussian_rasterization_amr import GaussianRasterizationSettings, GaussianRasterizer

from diff_gaussian_rasterization_amr import _RasterizeGaussians

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           starter=None,ender=None, starters=None, enders=None,
           interpolate_image = False):
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

    # print(raster_settings.debug)

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rawrasterizer = _RasterizeGaussians

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # if starter is not None:
    #     starter.record()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # for now manually set the centers of the 3 fovea steps to image center
    foveaCenters = torch.tensor([[viewpoint_camera.image_width/2, viewpoint_camera.image_height/2], 
                                  [viewpoint_camera.image_width/2, viewpoint_camera.image_height/2],
                                  [viewpoint_camera.image_width/2, viewpoint_camera.image_height/2],
                                  [viewpoint_camera.image_width/2, viewpoint_camera.image_height/2]], device='cuda')
    foveaRadii = torch.tensor([viewpoint_camera.image_width/2, 
                                 viewpoint_camera.image_width/4, 
                                 viewpoint_camera.image_width/8, 
                                 viewpoint_camera.image_width/16], device='cuda')

    # I record times here because the color and cov are not computed in python when testing fps
    # loading gaussian model parameters in python took similar time as rendering 
    # this is also what fov-3DGS does: Fov-3DGS/fov3dgs/gaussian_renderer_fov_mmfr/__init__.py:72


    # when using the raw rasterizer
    if shs is None:
        shs = torch.Tensor([])
    if colors_precomp is None:
        colors_precomp = torch.Tensor([])

    if scales is None:
        scales = torch.Tensor([])
    if rotations is None:
        rotations = torch.Tensor([])
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([])

    out_color_precomp = torch.Tensor([])



    if pipe.debug:
        print(" fovea step 0 ")
    foveaStep = 0
    buffered = False
    geomBuffer_precomp = torch.Tensor([]).to(torch.uint8)
    binningBuffer_precomp = torch.Tensor([]).to(torch.uint8)
    imageBuffer_precomp = torch.Tensor([]).to(torch.uint8)

    if starter is not None:
        starter.record()

    if starters is not None:
        starters[0].record()


    # radii = torch.empty(0, dtype=torch.int32, device='cuda')
    # parsed_point_list = torch.empty(0, dtype=torch.int32, device='cuda')
    # parsed_ranges = torch.empty(0, dtype=torch.int32, device='cuda')
    # parsed_tile_AMR_levels = torch.empty(0, dtype=torch.int32, device='cuda')
    # geomBuffer = torch.empty(0, dtype=torch.int8, device='cuda')
    # binningBuffer = torch.empty(0, dtype=torch.int8, device='cuda')
    # imgBuffer = torch.empty(0, dtype=torch.int8, device='cuda')

    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,
    #     foveaStep = int(0),
    #     out_color_precomp = None,
    #     # radii_precomp = None,
    #     # means2D_precomp = None,
    #     # conic_opacity_precomp = None,
    #     # geom_rgb_precomp = None,
    #     # point_list_precomp = None,
    #     # ranges_precomp = None,
    #     # tile_AMR_levels_last = None,
    #     # tile_AMR_levels_current = None,
    #     # geomBuffer_precomp = None,
    #     # binningBuffer_precomp = None,
    #     # imageBuffer_precomp = None,
    #     buffered = False,
    #     interpolate_image = interpolate_image
    # )

    # step 0: compute only the buffers
    rendered_image0, radii, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            False, # interpolate_image
        raster_settings,
    )
        

    
    if enders is not None:
        enders[0].record()

    if pipe.debug:
        reds = rendered_image0[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image0, "tmp0.png")

    # radii = radii.to(torch.int32)
    # parsed_point_list = parsed_point_list.to(torch.int32)
    # parsed_ranges = parsed_ranges.to(torch.int32)
    # parsed_tile_AMR_levels = parsed_tile_AMR_levels.to(torch.int32)

    # check all the dtypes that should be int
    # if pipe.debug:
        # print("radii dtype: ", radii.dtype)
        # print("parsed_point_list dtype: ", parsed_point_list.dtype)
        # print("parsed_ranges dtype: ", parsed_ranges.dtype)
        # print("parsed_tile_AMR_levels dtype: ", parsed_tile_AMR_levels.dtype)
        # print("geomBuffer dtype: ", geomBuffer.dtype)
        # print("binningBuffer dtype: ", binningBuffer.dtype)
        # print("imgBuffer dtype: ", imgBuffer.dtype)

    # make a deep copy of the torch tensor parsed_tile_AMR_levels and reduce by 3 (if <1, floor to 1) to get the AMR levels of step 0
    # parsed_tile_AMR_levels_step0 = parsed_tile_AMR_levels.clone()
    # parsed_tile_AMR_levels_step0 -= 3
    # parsed_tile_AMR_levels_step0[parsed_tile_AMR_levels_step0 < 1] = 1
    # # similarly for step 1, except -=2
    # parsed_tile_AMR_levels_step1 = parsed_tile_AMR_levels.clone()
    # parsed_tile_AMR_levels_step1 -= 2
    # parsed_tile_AMR_levels_step1[parsed_tile_AMR_levels_step1 < 1] = 1
    # TODO: if ourside the current fovea, set to same as last step



    foveaStep = 1
    out_color_precomp = rendered_image0 # should be all 0
    geomBuffer_precomp = geomBuffer
    binningBuffer_precomp = binningBuffer
    imageBuffer_precomp = imageBuffer
    if pipe.debug:
        reds = out_color_precomp[0].cpu().detach().numpy()
        redmask = reds == 0
        print("combined, Skipped pixels: ", np.sum(redmask))
        print("combined, Total pixels: ", redmask.size) 
    
    if pipe.debug:
        print(" fovea step 1 ")

    if starters is not None:
        starters[1].record()
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image1, _ = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,
    #     foveaStep = 1,
    #     out_color_precomp = rendered_image,
    #     # radii_precomp = radii,
    #     # means2D_precomp = parsed_means2D,
    #     # conic_opacity_precomp = parsed_conic_opacity,
    #     # geom_rgb_precomp = parsed_geom_rgb,
    #     # point_list_precomp = parsed_point_list,
    #     # ranges_precomp = parsed_ranges,
    #     # tile_AMR_levels_last = parsed_tile_AMR_levels_step0,
    #     # tile_AMR_levels_current = parsed_tile_AMR_levels_step1,
    #     # geomBuffer_precomp = geomBuffer,
    #     # binningBuffer_precomp = binningBuffer,
    #     # imageBuffer_precomp = imgBuffer,
    #     buffered = True,
    #     interpolate_image = interpolate_image
    # )

    # TODO: actually only imageBuffer changes (recording last and current levels)
    # step 1: compute the lowest quality
    rendered_image1, _, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            False, # interpolate_image
        raster_settings,
    )
        
    if enders is not None:
        enders[1].record()

    if pipe.debug:
        reds = rendered_image1[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image1, "tmp1.png")

        
    # similarly for step 2, except -=1
    # parsed_tile_AMR_levels_step2 = parsed_tile_AMR_levels.clone()
    # parsed_tile_AMR_levels_step2 -= 1
    # parsed_tile_AMR_levels_step2[parsed_tile_AMR_levels_step2 < 1] = 1



    foveaStep = 2
    buffered = True
    out_color_precomp = out_color_precomp + rendered_image1
    geomBuffer_precomp = geomBuffer
    binningBuffer_precomp = binningBuffer
    imageBuffer_precomp = imageBuffer
    if pipe.debug:
        reds = out_color_precomp[0].cpu().detach().numpy()
        redmask = reds == 0
        print("combined, Skipped pixels: ", np.sum(redmask))
        print("combined, Total pixels: ", redmask.size) 
    
    
    if pipe.debug:
        print(" fovea step 2 ")
    if starters is not None:
        starters[2].record()
    
    # rendered_image2, _, = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,
    #     foveaStep = 2,
    #     out_color_precomp = rendered_image1,
    #     # radii_precomp = radii,
    #     # means2D_precomp = parsed_means2D,
    #     # conic_opacity_precomp = parsed_conic_opacity,
    #     # geom_rgb_precomp = parsed_geom_rgb,
    #     # point_list_precomp = parsed_point_list,
    #     # ranges_precomp = parsed_ranges,
    #     # tile_AMR_levels_last = parsed_tile_AMR_levels_step1,
    #     # tile_AMR_levels_current = parsed_tile_AMR_levels_step2,
    #     # geomBuffer_precomp = geomBuffer1,
    #     # binningBuffer_precomp = binningBuffer1,
    #     # imageBuffer_precomp = imgBuffer1,
    #     buffered = True,
    #     interpolate_image = interpolate_image
    # )
    
    
    rendered_image2, _, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            False, # interpolate_image
        raster_settings,
    )

    if enders is not None:
        enders[2].record()

    
    if pipe.debug:
        reds = rendered_image2[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image2, "tmp2.png")


    # step 3 is the same as parsed_tile_AMR_levels
    # parsed_tile_AMR_levels_step3 = parsed_tile_AMR_levels.clone()


    foveaStep = 3
    out_color_precomp = out_color_precomp + rendered_image2
    geomBuffer_precomp = geomBuffer
    binningBuffer_precomp = binningBuffer
    imageBuffer_precomp = imageBuffer
    if pipe.debug:
        reds = out_color_precomp[0].cpu().detach().numpy()
        redmask = reds == 0
        print("combined, Skipped pixels: ", np.sum(redmask))
        print("combined, Total pixels: ", redmask.size) 
    
    
    if pipe.debug:
        print(" fovea step 3 ")
    if starters is not None:
        starters[3].record()

    # rendered_image3, _ = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,
    #     foveaStep = 3,
    #     out_color_precomp = rendered_image2,
    #     # radii_precomp = radii,
    #     # means2D_precomp = parsed_means2D,
    #     # conic_opacity_precomp = parsed_conic_opacity,
    #     # geom_rgb_precomp = parsed_geom_rgb,
    #     # point_list_precomp = parsed_point_list,
    #     # ranges_precomp = parsed_ranges,
    #     # tile_AMR_levels_last = parsed_tile_AMR_levels_step2,
    #     # tile_AMR_levels_current = parsed_tile_AMR_levels_step3,
    #     # geomBuffer_precomp = geomBuffer2,
    #     # binningBuffer_precomp = binningBuffer2,
    #     # imageBuffer_precomp = imgBuffer2,
    #     buffered = True,
    #     interpolate_image = interpolate_image
    # )
    
    
    rendered_image3, _, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            False, # interpolate_image
        raster_settings,
    )
    

    if enders is not None:
        enders[3].record()


    if pipe.debug:
        reds = rendered_image3[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image3, "tmp3.png")


    foveaStep = 4
    out_color_precomp = out_color_precomp + rendered_image3
    geomBuffer_precomp = geomBuffer
    binningBuffer_precomp = binningBuffer
    imageBuffer_precomp = imageBuffer
    if pipe.debug:
        reds = out_color_precomp[0].cpu().detach().numpy()
        redmask = reds == 0
        print("combined, Skipped pixels: ", np.sum(redmask))
        print("combined, Total pixels: ", redmask.size) 
    
    if pipe.debug:
        print(" fovea step 4 ")

    if starters is not None:
        starters[4].record()

    # rendered_image3, _ = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,
    #     foveaStep = 3,
    #     out_color_precomp = rendered_image2,
    #     # radii_precomp = radii,
    #     # means2D_precomp = parsed_means2D,
    #     # conic_opacity_precomp = parsed_conic_opacity,
    #     # geom_rgb_precomp = parsed_geom_rgb,
    #     # point_list_precomp = parsed_point_list,
    #     # ranges_precomp = parsed_ranges,
    #     # tile_AMR_levels_last = parsed_tile_AMR_levels_step2,
    #     # tile_AMR_levels_current = parsed_tile_AMR_levels_step3,
    #     # geomBuffer_precomp = geomBuffer2,
    #     # binningBuffer_precomp = binningBuffer2,
    #     # imageBuffer_precomp = imgBuffer2,
    #     buffered = True,
    #     interpolate_image = interpolate_image
    # )
    
    
    rendered_image4, _, geomBuffer, binningBuffer, imageBuffer = rawrasterizer.apply(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
            foveaStep,
            out_color_precomp,
            # radii_precomp,
            # means2D_precomp,
            # conic_opacity_precomp,
            # geom_rgb_precomp,
            # point_list_precomp,
            # ranges_precomp,
            # tile_AMR_levels_last,
            # tile_AMR_levels_current,
            geomBuffer_precomp,
            binningBuffer_precomp,
            imageBuffer_precomp,
            interpolate_image,
        raster_settings,
    )
    
    if ender is not None:
        ender.record()

    if enders is not None:
        enders[4].record()

    
    if pipe.debug:
        reds = rendered_image4[0].cpu().detach().numpy()
        redmask = reds == 0
        print("Skipped pixels: ", np.sum(redmask))
        print("Total pixels: ", redmask.size) 
        torchvision.utils.save_image(rendered_image4, "tmp4.png")


    out_color_precomp = out_color_precomp + rendered_image4

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": out_color_precomp,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            # "means2D": parsed_means2D,
            # "conic_opacity": parsed_conic_opacity,
            # "geom_rgb": parsed_geom_rgb,
            # "point_list": parsed_point_list,
            # "ranges": parsed_ranges,
            # "tile_AMR_levels": parsed_tile_AMR_levels
            }


# render the entire scene once without foveation
def render_once(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           starter=None,ender=None):
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

    # print(raster_settings.debug)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # if starter is not None:
    #     starter.record()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # I record times here because the color and cov are not computed in python when testing fps
    # loading gaussian model parameters in python took similar time as rendering 
    # this is also what fov-3DGS does: Fov-3DGS/fov3dgs/gaussian_renderer_fov_mmfr/__init__.py:72



    # radii = torch.empty(0, dtype=torch.int32, device='cuda')
    # parsed_point_list = torch.empty(0, dtype=torch.int32, device='cuda')
    # parsed_ranges = torch.empty(0, dtype=torch.int32, device='cuda')
    # parsed_tile_AMR_levels = torch.empty(0, dtype=torch.int32, device='cuda')

    
    if starter is not None:
        starter.record()

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, parsed_means2D, parsed_conic_opacity, parsed_geom_rgb, parsed_point_list, parsed_ranges, parsed_tile_AMR_levels, _, _, _= rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp,
    #     foveaStep = int(-2),
    #     out_color_precomp = None,
    #     radii_precomp = None,
    #     means2D_precomp = None,
    #     conic_opacity_precomp = None,
    #     geom_rgb_precomp = None,
    #     point_list_precomp = None,
    #     ranges_precomp = None,
    #     tile_AMR_levels_last = None,
    #     tile_AMR_levels_current = None,
    #     geomBuffer_precomp = None,
    #     binningBuffer_precomp = None,
    #     imageBuffer_precomp = None
    # )

    rendered_image, radii, _, _, _= rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        foveaStep = int(-2)
    )
        
    if ender is not None:
        ender.record()


    


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }