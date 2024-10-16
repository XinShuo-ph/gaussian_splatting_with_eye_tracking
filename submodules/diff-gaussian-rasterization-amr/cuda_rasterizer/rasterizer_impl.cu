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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// helper function to compute percentile to determine the AMR level
template<typename T>
T calculateAMRbyPercentileRadixSort(T* d_array, int size, float percentile, void* d_temp_storage, size_t& temp_storage_bytes, bool debug)
{
    // Allocate temporary storage for sorting
    if (d_temp_storage == nullptr)
    {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes), debug);
    }

    // Sort the array
    CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_array, d_array, size), debug);

    // Calculate the index for the desired percentile
    int percentile_index = static_cast<int>(percentile * size);

    // Copy the result back to host
    T result;
    CHECK_CUDA(cudaMemcpy(&result, d_array + percentile_index, sizeof(T), cudaMemcpyDeviceToHost), debug);

    return result;
}

// Kernel to calculate n_intersections
__global__ void calculateIntersections(uint2* ranges, uint32_t* n_intersections, int num_tiles)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx < num_tiles)
	{
		n_intersections[idx] = ranges[idx].y - ranges[idx].x;
	}
}
// Add this kernel at the top of the file or in a separate header
__global__ void setAMRLevelsKernel(uint32_t* n_intersections, uint32_t* percentile_values, uint32_t* tile_AMR_levels, int num_tiles)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx < num_tiles)
    {
        uint32_t value = n_intersections[idx];
        if (value <= percentile_values[0])
            tile_AMR_levels[idx] = 1;
        else if (value <= percentile_values[1])
            tile_AMR_levels[idx] = 2;
        else if (value <= percentile_values[2])
            tile_AMR_levels[idx] = 3;
        else
            tile_AMR_levels[idx] = 4;
    }
}

// shall pass the fovea radii and centers of current step in the future
__global__ void setFoveaAMRLevelsKernel(int foveaStep, 
	uint32_t* tile_AMR_levels_last, uint32_t* tile_AMR_levels_current, uint32_t* tile_AMR_levels, int num_tiles)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx < num_tiles)
    {
        uint32_t current_level = tile_AMR_levels[idx]; // the AMR level for fully rendered image
        
        switch(foveaStep)
        {
			case 0:
				break; // Do nothing, only do preprocessing for step 0
			case 1:
				tile_AMR_levels_last[idx] = 0;
				tile_AMR_levels_current[idx] = ((current_level >= 1) ? 1 : tile_AMR_levels_last[idx]);
				break;
            case 2:
                tile_AMR_levels_last[idx] = tile_AMR_levels_current[idx];
                tile_AMR_levels_current[idx] = ((current_level >= 2) ? 2 : tile_AMR_levels_last[idx]);
                break;
            case 3:
                tile_AMR_levels_last[idx] = tile_AMR_levels_current[idx];
                tile_AMR_levels_current[idx] = ((current_level >= 3) ? 3 : tile_AMR_levels_last[idx]);
                break;
            case 4:
                tile_AMR_levels_last[idx] = tile_AMR_levels_current[idx];
                tile_AMR_levels_current[idx] = ((current_level >= 4) ? 4 : tile_AMR_levels_last[idx]);
                break;
            default:
                // if <0, set full render
				tile_AMR_levels_last[idx] = 0;
				tile_AMR_levels_current[idx] = current_level;
                break;
        }
    }
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);

	// added for AMR and debug purposes
	obtain(chunk, img.n_intersections, N, 128);
    obtain(chunk, img.n_intersections_sorted, N, 128);
    obtain(chunk, img.tile_AMR_levels, N, 128);
	obtain(chunk, img.tile_AMR_levels_last, N, 128);
	obtain(chunk, img.tile_AMR_levels_current, N, 128);
    
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	const int foveaStep,
	const float* out_color_precomp,
	// const int* radii_precomp,
	// const float* means2D_precomp,
	// const float* conic_opacity_precomp,
	// const float* geom_rgb_precomp,
	// const int* point_list_precomp,
	// const int* ranges_precomp,
	// const int* tile_AMR_levels_last,
	// const int* tile_AMR_levels_current,
	char* geom_buffer, // these three are precomputed buffers
	char* binning_buffer,
	char* image_buffer,
	float* out_color,
	int* radii,
	const bool interpolate_image,
	bool debug)
{
	if (foveaStep >= 1 ){
		if (debug) {
			std::cout << "CudaRasterizer::Rasterizer::forward() Foveated rendering step: " << foveaStep << std::endl;
		}

		
		GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
		// Retrieve total number of Gaussian instances to launch and resize aux buffers
		int num_rendered;
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
		if (debug){
			std::cout << "CudaRasterizer::Rasterizer::forward() num_rendered: " << num_rendered << " in foveaStep: " << foveaStep << std::endl;
		}
		BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
		ImageState imgState = ImageState::fromChunk(image_buffer, width * height);

		dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
		int num_tiles = tile_grid.x * tile_grid.y;
		dim3 render_tile_grid( ( (width + BLOCK_X - 1) / BLOCK_X ) * RENDER_BLOCK_RATIO, ( (height + BLOCK_Y - 1) / BLOCK_Y ) * RENDER_BLOCK_RATIO, 1);
		// dim3 block(BLOCK_X, BLOCK_Y, 1);
		dim3 block_for_render(RENDER_BLOCK_X, RENDER_BLOCK_Y, 1);
		// std::cout << "Render tile grid: " << render_tile_grid.x << "x" << render_tile_grid.y << std::endl;
		// std::cout << "Tile grid: " << tile_grid.x << "x" << tile_grid.y << std::endl;

		// set AMR levels for this fovea step
		if (debug){
			std::cout << "CudaRasterizer::Rasterizer::forward() setFoveaAMRLevelsKernel" << std::endl;
		}
		setFoveaAMRLevelsKernel<<<(num_tiles + 255) / 256, 256>>>(foveaStep, imgState.tile_AMR_levels_last, imgState.tile_AMR_levels_current, imgState.tile_AMR_levels, num_tiles);
		CHECK_CUDA(, debug)
		if (debug){
			// Allocate host memory
			uint32_t host_last, host_current, host_actual;
			
			// Copy first element of each array from device to host
			cudaMemcpy(&host_last, imgState.tile_AMR_levels_last, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaMemcpy(&host_current, imgState.tile_AMR_levels_current, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaMemcpy(&host_actual, imgState.tile_AMR_levels, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			
			// Output the values
			std::cout << "CudaRasterizer::Rasterizer::forward() imgState.tile_AMR_levels_last[0]: " << host_last 
					<< " imgState.tile_AMR_levels_current[0]: " << host_current 
					<< " imgState.tile_AMR_levels[0]: " << host_actual << std::endl;
		}
		// use precomputed values to directly run render
		const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

		// similarly set the ptr to the precomputed values 
		const float* out_color_precomp_ptr = out_color_precomp;
		// const float* out_color_precomp_ptr = out_color_precomp != nullptr ? out_color_precomp : out_color;	// these are not set when foveaStep <=0, still need to pass them
	
		// const int* radii_precomp_ptr = radii_precomp;
		// const float* means2D_precomp_ptr = means2D_precomp;
		// const float* conic_opacity_precomp_ptr = conic_opacity_precomp;
		// const float* geom_rgb_precomp_ptr = geom_rgb_precomp;
		// const int* point_list_precomp_ptr = point_list_precomp;
		// const int* ranges_precomp_ptr = ranges_precomp;
		// const int* tile_AMR_levels_last_ptr = tile_AMR_levels_last;
		// const int* tile_AMR_levels_current_ptr = tile_AMR_levels_current;

		if (debug) {
			std::cout << "CudaRasterizer::Rasterizer::forward() render" << std::endl;
		}

		CHECK_CUDA(FORWARD::render(
			render_tile_grid, block_for_render,
			imgState.ranges, imgState.tile_AMR_levels_current,
			// ranges_precomp_ptr, tile_AMR_levels_current_ptr,
			binningState.point_list,
			// point_list_precomp_ptr,
			width, height,
			geomState.means2D,
			// means2D_precomp_ptr,
			feature_ptr,
			geomState.conic_opacity,
			// conic_opacity_precomp_ptr,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			foveaStep, // new foveated rendering steps
			out_color_precomp_ptr,
			imgState.tile_AMR_levels_last,
			interpolate_image
			), debug)

/*
		
		// placeholder for legacy buffers, should remove them in the future
		size_t chunk_size = required<GeometryState>(P);
		char* chunkptr = geometryBuffer(chunk_size);
		GeometryState geomState_legacy = GeometryState::fromChunk(chunkptr, P);
		size_t img_chunk_size = required<ImageState>(width * height);
		char* img_chunkptr = imageBuffer(img_chunk_size);
		ImageState imgState_legacy = ImageState::fromChunk(img_chunkptr, width * height);
		size_t binning_chunk_size = required<BinningState>(num_rendered);
		char* binning_chunkptr = binningBuffer(binning_chunk_size);
		BinningState binningState_legacy = BinningState::fromChunk(binning_chunkptr, num_rendered);

		// Copy the buffers used to legacy buffers, this may be a bit slow, should change to a better way in the future
		CHECK_CUDA(cudaMemcpy(geomState_legacy.depths, geomState.depths, P * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.clamped, geomState.clamped, P * 3 * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.internal_radii, geomState.internal_radii, P * sizeof(int), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.means2D, geomState.means2D, P * sizeof(float2), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.cov3D, geomState.cov3D, P * 6 * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.conic_opacity, geomState.conic_opacity, P * sizeof(float4), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.rgb, geomState.rgb, P * 3 * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.tiles_touched, geomState.tiles_touched, P * sizeof(int), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(geomState_legacy.point_offsets, geomState.point_offsets, P * sizeof(int), cudaMemcpyDeviceToDevice), debug);

		CHECK_CUDA(cudaMemcpy(imgState_legacy.accum_alpha, imgState.accum_alpha, width * height * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.n_contrib, imgState.n_contrib, width * height * sizeof(int), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.ranges, imgState.ranges, width * height * sizeof(uint2), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.n_intersections, imgState.n_intersections, width * height * sizeof(uint32_t), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.n_intersections_sorted, imgState.n_intersections_sorted, width * height * sizeof(uint32_t), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.tile_AMR_levels, imgState.tile_AMR_levels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.tile_AMR_levels_last, imgState.tile_AMR_levels_last, width * height * sizeof(uint32_t), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(imgState_legacy.tile_AMR_levels_current, imgState.tile_AMR_levels_current, width * height * sizeof(uint32_t), cudaMemcpyDeviceToDevice), debug);

		CHECK_CUDA(cudaMemcpy(binningState_legacy.point_list, binningState.point_list, num_rendered * sizeof(int), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(binningState_legacy.point_list_unsorted, binningState.point_list_unsorted, num_rendered * sizeof(int), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(binningState_legacy.point_list_keys, binningState.point_list_keys, num_rendered * sizeof(uint64_t), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(binningState_legacy.point_list_keys_unsorted, binningState.point_list_keys_unsorted, num_rendered * sizeof(uint64_t), cudaMemcpyDeviceToDevice), debug);
*/
		
		// // prepare for the next step, set tile_AMR_levels_last to 1, and reduce tile_AMR_levels by 2 (if >2, else set to 1)
		// setFoveaAMRLevelsKernel<<<(num_tiles + 255) / 256, 256>>>(foveaStep, imgState.tile_AMR_levels_last, imgState.tile_AMR_levels_current, imgState.tile_AMR_levels, num_tiles);
		return num_rendered;
	}

	// else, we still need preprocess
	if (debug) {
		std::cout << "CudaRasterizer::Rasterizer::forward() Regular rendering" << std::endl;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// dim3 test_tile_grid((tile_grid.x) * RENDER_BLOCK_RATIO, (tile_grid.y) * RENDER_BLOCK_RATIO, 1);
	// std::cout << "Tile grid: " << tile_grid.x << "x" << tile_grid.y << std::endl;
	// std::cout << "Test tile grid: " << test_tile_grid.x << "x" << test_tile_grid.y << std::endl;
	dim3 render_tile_grid( ( (width + BLOCK_X - 1) / BLOCK_X ) * RENDER_BLOCK_RATIO, ( (height + BLOCK_Y - 1) / BLOCK_Y ) * RENDER_BLOCK_RATIO, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	dim3 block_for_render(RENDER_BLOCK_X, RENDER_BLOCK_Y, 1);
	// std::cout << "Render tile grid: " << render_tile_grid.x << "x" << render_tile_grid.y << std::endl;
	// std::cout << "Tile grid: " << tile_grid.x << "x" << tile_grid.y << std::endl;

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	if (debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() Preprocessing" << std::endl;
	}
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	if (debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() num_rendered: " << num_rendered << " in foveaStep: " << foveaStep << std::endl;
	}

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	if (debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() duplicateWithKeys" << std::endl;
	}
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	if(debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() SortPairs" << std::endl;
	}
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	if(debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() identifyTileRanges" << std::endl;
	}
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// determine AMR levels by percentiles
	// Calculate n_intersections
	int num_tiles = tile_grid.x * tile_grid.y;
	// uint32_t* n_intersections;
	// uint32_t* n_intersections_sorted;
	// CHECK_CUDA(cudaMalloc(&n_intersections, num_tiles * sizeof(uint32_t)), debug);
	// CHECK_CUDA(cudaMalloc(&n_intersections_sorted, num_tiles * sizeof(uint32_t)), debug);

	if(debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() calculateIntersections" << std::endl;
	}
	// Launch kernel for n_intersections
	// calculateIntersections<<<(num_tiles + 255) / 256, 256>>>(imgState.ranges, n_intersections, num_tiles);
	calculateIntersections<<<(num_tiles + 255) / 256, 256>>>(imgState.ranges, imgState.n_intersections, num_tiles);
	CHECK_CUDA(, debug);


	// get percentile values and set AMR levels
	// should change to general numbers percentiles[AMR_MAX_LEVELS - 1]
	float percentiles[3] = {0.25f, 0.5f, 0.9f};
	uint32_t percentile_values[3];
	// uint32_t* tile_AMR_levels; // use uint8_t to save memory (is this necessary?)
	// CHECK_CUDA(cudaMalloc(&tile_AMR_levels, num_tiles * sizeof(uint32_t)), debug);

	// Allocate temporary storage for sorting
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	// when d_temp_storage is nullptr, cub::DeviceRadixSort::SortKeys will return the required size of temp_storage_bytes
	// CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, n_intersections, n_intersections_sorted, num_tiles), debug);
	CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, imgState.n_intersections, imgState.n_intersections_sorted, num_tiles), debug);
	CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes), debug);

	// Sort the array
	// CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, n_intersections, n_intersections_sorted, num_tiles), debug);
	CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, imgState.n_intersections, imgState.n_intersections_sorted, num_tiles), debug);

	// Calculate percentile indices
	int percentile_indices[3];
	for (int i = 0; i < 3; i++)
	{
		percentile_indices[i] = static_cast<int>(percentiles[i] * num_tiles);
	}

	// Copy percentile values back to host
	for (int i = 0; i < 3; i++) {
		// CHECK_CUDA(cudaMemcpy(&percentile_values[i], n_intersections_sorted + percentile_indices[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaMemcpy(&percentile_values[i], imgState.n_intersections_sorted + percentile_indices[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
	}

	if(debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() setAMRLevelsKernel" << std::endl;
	}
	// Set AMR levels
	// setAMRLevelsKernel<<<(num_tiles + 255) / 256, 256>>>(n_intersections, percentile_values, tile_AMR_levels, num_tiles);
	setAMRLevelsKernel<<<(num_tiles + 255) / 256, 256>>>(imgState.n_intersections, percentile_values, imgState.tile_AMR_levels, num_tiles);
	CHECK_CUDA(, debug);

	// Clean up
	cudaFree(d_temp_storage);

	// step 0: only preprocess
	if (foveaStep == 0) {
		return num_rendered;
	}
	// setting foveaStep = -1 will do rendering normally

	setFoveaAMRLevelsKernel<<<(num_tiles + 255) / 256, 256>>>(foveaStep, imgState.tile_AMR_levels_last, imgState.tile_AMR_levels_current, imgState.tile_AMR_levels, num_tiles);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* out_color_precomp_ptr = out_color_precomp != nullptr ? out_color_precomp : out_color;	// these are not set when foveaStep <=0, still need to pass them
	
	if(debug){
		std::cout << "CudaRasterizer::Rasterizer::forward() render" << std::endl;
	}
	CHECK_CUDA(FORWARD::render(
		render_tile_grid, block_for_render,
		imgState.ranges, imgState.tile_AMR_levels,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		foveaStep, // new foveated rendering steps
		out_color_precomp_ptr,
		imgState.tile_AMR_levels_last,
		interpolate_image), debug)

	// if (foveaStep == 0 ) {
	// 	if(debug){
	// 		std::cout << "CudaRasterizer::Rasterizer::forward() setFoveaAMRLevelsKernel" << std::endl;
	// 	}
	// 	// prepare for the next step, set tile_AMR_levels_last to 1, and reduce tile_AMR_levels by 2 (if >2, else set to 1)
	// 	setFoveaAMRLevelsKernel<<<(num_tiles + 255) / 256, 256>>>(foveaStep, imgState.tile_AMR_levels_last, imgState.tile_AMR_levels_current, imgState.tile_AMR_levels, num_tiles);
	// }

	if(debug){
			std::cout << "CudaRasterizer::Rasterizer::forward() return" << std::endl;
		}
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 render_tile_grid(tile_grid.x * RENDER_BLOCK_RATIO, tile_grid.y * RENDER_BLOCK_RATIO, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 block_for_render(RENDER_BLOCK_X, RENDER_BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		render_tile_grid,
		block_for_render,
		// tile_grid,
		// block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}