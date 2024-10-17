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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
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
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
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

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
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
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
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
            raster_settings.debug
        )

        # turn off the original debug for now

        # # Invoke C++/CUDA rasterizer
        # if raster_settings.debug:
        #     cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        #     try:
        #         num_rendered, color, radii, parsed_means2D, parsed_conic_opacity, parsed_geom_rgb, parsed_point_list, parsed_ranges, parsed_tile_AMR_levels, geomBuffer, binningBuffer, imgBuffer= _C.rasterize_gaussians(*args)
        #     except Exception as ex:
        #         torch.save(cpu_args, "snapshot_fw.dump")
        #         print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
        #         raise ex
        # else:
        #     num_rendered, color, radii, parsed_means2D, parsed_conic_opacity, parsed_geom_rgb, parsed_point_list, parsed_ranges, parsed_tile_AMR_levels, geomBuffer, binningBuffer, imgBuffer= _C.rasterize_gaussians(*args)
        
        
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer= _C.rasterize_gaussians(*args)



        # # Ensure correct types without conversion
        # if raster_settings.debug:
        #     # assert radii.dtype == torch.int32
        #     # assert parsed_point_list.dtype == torch.int32
        #     # assert parsed_ranges.dtype == torch.int32
        #     # assert parsed_tile_AMR_levels.dtype == torch.int32
        #     # assert geomBuffer.dtype == torch.int8
        #     # assert binningBuffer.dtype == torch.int8
        #     # assert imgBuffer.dtype == torch.int8
        #     # print the dtype
        #     print("radii.dtype: ", radii.dtype)
        #     print("parsed_point_list.dtype: ", parsed_point_list.dtype)
        #     print("parsed_ranges.dtype: ", parsed_ranges.dtype)
        #     print("parsed_tile_AMR_levels.dtype: ", parsed_tile_AMR_levels.dtype)
        #     print("geomBuffer.dtype: ", geomBuffer.dtype)
        #     print("binningBuffer.dtype: ", binningBuffer.dtype)
        #     print("imgBuffer.dtype: ", imgBuffer.dtype)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, geomBuffer, binningBuffer, imgBuffer

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None,
                foveaStep = int(0),
                out_color_precomp = None,
                # radii_precomp = None,
                # means2D_precomp = None,
                # conic_opacity_precomp = None,
                # geom_rgb_precomp = None,
                # point_list_precomp = None,
                # ranges_precomp = None,
                # tile_AMR_levels_last = None,
                # tile_AMR_levels_current = None,
                geomBuffer_precomp = None,
                binningBuffer_precomp = None,
                imageBuffer_precomp = None,
                interpolate_image = True):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
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

        if out_color_precomp is None:
            out_color_precomp = torch.Tensor([])
        # if radii_precomp is None:
        #     radii_precomp = torch.Tensor([]).to(torch.int32)
        # if means2D_precomp is None:
        #     means2D_precomp = torch.Tensor([])
        # if conic_opacity_precomp is None:
        #     conic_opacity_precomp = torch.Tensor([])
        # if geom_rgb_precomp is None:
        #     geom_rgb_precomp = torch.Tensor([])
        # if point_list_precomp is None:
        #     point_list_precomp = torch.Tensor([]).to(torch.int32)
        # if ranges_precomp is None:
        #     ranges_precomp = torch.Tensor([]).to(torch.int32)
        # if tile_AMR_levels_last is None:
        #     tile_AMR_levels_last = torch.Tensor([]).to(torch.int32)
        # if tile_AMR_levels_current is None:
        #     tile_AMR_levels_current = torch.Tensor([]).to(torch.int32)
        if geomBuffer_precomp is None:
            geomBuffer_precomp = torch.Tensor([]).to(torch.uint8)
        if binningBuffer_precomp is None:
            binningBuffer_precomp = torch.Tensor([]).to(torch.uint8)
        if imageBuffer_precomp is None:
            imageBuffer_precomp = torch.Tensor([]).to(torch.uint8)
            # set the buffer type to char (int8?)
        #     geomBuffer_precomp = geomBuffer_precomp.to(torch.int8)
        #     binningBuffer_precomp = binningBuffer_precomp.to(torch.int8)
        #     imageBuffer_precomp = imageBuffer_precomp.to(torch.int8)

        # foveaStep = int(foveaStep)
        # radii_precomp = radii_precomp.to(torch.int32)
        # point_list_precomp = point_list_precomp.to(torch.int32)
        # ranges_precomp = ranges_precomp.to(torch.int32)
        # tile_AMR_levels_last = tile_AMR_levels_last.to(torch.int32)
        # tile_AMR_levels_current = tile_AMR_levels_current.to(torch.int32)


        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
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

