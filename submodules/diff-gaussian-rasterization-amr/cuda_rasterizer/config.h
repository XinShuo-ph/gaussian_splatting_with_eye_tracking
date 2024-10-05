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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
// #define TILE_LEVEL 5 // BLOCK_X should be 2^TILE_LEVEL
#define BLOCK_X 32
#define BLOCK_Y 32
// render each block by 16x16 pixels four times
// 16*16 threads per cooperative group seems to be optimal
#define RENDER_BLOCK_RATIO 2
#define RENDER_BLOCK_X 16 // should be BLOCK_X / RENDER_BLOCK_RATIO
#define RENDER_BLOCK_Y 16
#define AMR_MAX_LEVELS 4 // should be RENDER_BLOCK_RATIO^2

#endif