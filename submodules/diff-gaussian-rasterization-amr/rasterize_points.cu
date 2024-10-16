/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// save 5 intermediate quantities now
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ParseBuffers(
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imgBuffer,
    int P, int width, int height)
{
    char* geom_ptr = reinterpret_cast<char*>(geomBuffer.data_ptr());
    char* binning_ptr = reinterpret_cast<char*>(binningBuffer.data_ptr());
    char* img_ptr = reinterpret_cast<char*>(imgBuffer.data_ptr());

    CudaRasterizer::GeometryState geom = CudaRasterizer::GeometryState::fromChunk(geom_ptr, P);
    CudaRasterizer::BinningState binning = CudaRasterizer::BinningState::fromChunk(binning_ptr, P);
    CudaRasterizer::ImageState img = CudaRasterizer::ImageState::fromChunk(img_ptr, width * height);

    // Create tensors from the parsed data
    torch::Tensor means2D = torch::from_blob(geom.means2D, {P, 2}, geomBuffer.options().dtype(torch::kFloat32));
    torch::Tensor conic_opacity = torch::from_blob(geom.conic_opacity, {P, 4}, geomBuffer.options().dtype(torch::kFloat32));
    torch::Tensor geom_rgb = torch::from_blob(geom.rgb, {P*3}, geomBuffer.options().dtype(torch::kFloat32));
    torch::Tensor point_list = torch::from_blob(binning.point_list, {P}, binningBuffer.options().dtype(torch::kInt32));
	int grid_x = (width + BLOCK_X - 1) / BLOCK_X;
	int grid_y =  (height + BLOCK_Y - 1) / BLOCK_Y;
    torch::Tensor ranges = torch::from_blob(img.ranges, {grid_x*grid_y, 2}, imgBuffer.options().dtype(torch::kInt32));
    torch::Tensor tile_AMR_levels = torch::from_blob(img.tile_AMR_levels, {grid_x*grid_y}, imgBuffer.options().dtype(torch::kInt32));

    return std::make_tuple(means2D, conic_opacity, geom_rgb, point_list, ranges, tile_AMR_levels);
}


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const int foveaStep, // =-1 means no foveation, =0,1,2,3 corresponds to progressively higher quality
	const torch::Tensor& out_color_precomp, // precomputed color (from last step)
	// const torch::Tensor& radii_precomp, // precomputed radii
	// const torch::Tensor& means2D_precomp, // precomputed means2D 
	// const torch::Tensor& conic_opacity_precomp, // precomputed conic_opacity
	// const torch::Tensor& geom_rgb_precomp, // precomputed geom_rgb
	// const torch::Tensor& point_list_precomp, // precomputed point_list
	// const torch::Tensor& ranges_precomp, // precomputed ranges
	// const torch::Tensor& tile_AMR_levels_last, // AMR levels of the last step
	// const torch::Tensor& tile_AMR_levels_current, // AMR levels of the current step
	const torch::Tensor& geomBuffer_precomp, // pass the buffer, this this can work, we do not need things above
	const torch::Tensor& binningBuffer_precomp,
	const torch::Tensor& imageBuffer_precomp,
	const bool interpolate_image, // whether to interpolate the image or leave unrendered blank
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  if (debug) {
	std::cout << "RasterizeGaussiansCUDA" << std::endl;
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }
		if (debug) {
			std::cout << "RasterizeGaussiansCUDA: forward" << std::endl;
		}
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		foveaStep,
		out_color_precomp.contiguous().data<float>(),
		// radii_precomp.contiguous().data<int>(),
		// means2D_precomp.contiguous().data<float>(),
		// conic_opacity_precomp.contiguous().data<float>(),
		// geom_rgb_precomp.contiguous().data<float>(),
		// point_list_precomp.contiguous().data<int>(),
		// ranges_precomp.contiguous().data<int>(),
		// tile_AMR_levels_last.contiguous().data<int>(),
		// tile_AMR_levels_current.contiguous().data<int>(),
		reinterpret_cast<char*>(geomBuffer_precomp.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer_precomp.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer_precomp.contiguous().data_ptr()),
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		interpolate_image,
		debug);
		if (debug) {
			std::cout << "RasterizeGaussiansCUDA: forward done" << std::endl;
		}
  }
//   return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
    if (foveaStep > 0){ // if entered foveated rendering, use precomputed buffers
		// if (debug) {
		// 	std::cout << "RasterizeGaussiansCUDA: ParseBuffers" << std::endl;
		// }
		// auto [parsed_means2D, parsed_conic_opacity, parsed_geom_rgb, parsed_point_list, parsed_ranges, parsed_tile_AMR_levels] = ParseBuffers(geomBuffer_precomp, binningBuffer_precomp, imageBuffer_precomp, P, W, H);	
		if (debug) {
			std::cout << "RasterizeGaussiansCUDA: return" << std::endl;
		}
		return std::make_tuple(rendered, out_color, radii, geomBuffer_precomp, binningBuffer_precomp, imageBuffer_precomp);

	}
	// otherwise, same as standard 3DGS
	// if (debug) {
	// 	std::cout << "RasterizeGaussiansCUDA: ParseBuffers" << std::endl;
	// }
	// auto [parsed_means2D, parsed_conic_opacity, parsed_geom_rgb, parsed_point_list, parsed_ranges, parsed_tile_AMR_levels] = ParseBuffers(geomBuffer, binningBuffer, imgBuffer, P, W, H);
	if (debug) {
		std::cout << "RasterizeGaussiansCUDA: return" << std::endl;
	}
	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}