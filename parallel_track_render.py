from mpi4py import MPI
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import time

# from gaussian_renderer_amr import render, GaussianModel

# write render () function on my own
import math
from diff_gaussian_rasterization_amr import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_amr import _RasterizeGaussians
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from fovealnet.timm_vit import VisionTransformer

def load_image(image_path):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

pix_x = 1920
pix_y = 1080

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

# Add arguments from track.py
parser.add_argument("--foveal_model_path", default="/home/ubuntu/gaussian_splatting_with_eye_tracking/fovealnet/results_20241022/model_minmax_0.8.pt", type=str)
parser.add_argument("--eye_image_folder", default="/home/ubuntu/openeds/test/sequences/0000/", type=str)
parser.add_argument("--eye_image_sequence_folder", default="/home/ubuntu/openeds/test/sequences/", type=str)
parser.add_argument("--eye_image_sequence_id_start", default=0, type=int)
parser.add_argument("--eye_image_sequence_id_end", default=100, type=int)
parser.add_argument("--foveal_output_file", type=str, default="predictions.txt")
parser.add_argument("--foveal_layer_timer", action="store_true")

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--gaze_r2", default=600, type=float)
parser.add_argument("--gaze_r3", default=400, type=float)
parser.add_argument("--gaze_r4", default=200, type=float)
parser.add_argument("--test_no_render_laststep", action="store_true") # test the time of purely passing the data in foveastep 4
args = get_combined_args(parser)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create shared memory window for gaze predictions
# if rank == 0:
#     # Rank 0 creates the shared memory
#     sync_gaze_prediction = np.zeros(2, dtype=np.float32)
# else:
#     # Other ranks don't need local buffer
#     sync_gaze_prediction = np.zeros(2, dtype=np.float32)

sync_gaze_prediction = np.zeros(2, dtype=np.float32)
win = MPI.Win.Create(sync_gaze_prediction, comm=comm)

gaze_predictions_buffer = np.zeros((150, 2), dtype=np.float32) # at most 150 gaze for each head position
fovealnet_level_buffer = np.zeros(150, dtype=np.int32)

win1 = MPI.Win.Create(gaze_predictions_buffer, comm=comm)
win2 = MPI.Win.Create(fovealnet_level_buffer, comm=comm)


if rank == 0:  # Gaussian Splatting process

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    print(f"Gaussian Splatting process is using {num_sms} stream multiprocessors.")

    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    mydataset = model.extract(args)

    gaussians = GaussianModel(mydataset.sh_degree)
    scene = Scene(mydataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()
    views = [views[i] for i in range(100)]
    bg_color = [1,1,1] if mydataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ratio = 1

    print(f"Rendering at {int(pix_x*ratio)}x{int(pix_y*ratio)}")
    # change the image width and height
    for i in range(len(views)):
        views[i].image_width = int(pix_x*ratio)
        views[i].image_height = int(pix_y*ratio)


    # test fps by counting time
    starters = []
    enders = []
    starters1 = []
    enders1 = []
    for _, _ in enumerate(views):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starters.append(starter)
        enders.append(ender)
        starters1.append(starter1)
        enders1.append(ender1)

    # Create zero tensor for screenspace points
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Get Gaussian parameters
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    # Setup for covariance/scales/rotations
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features
    


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # Wait for FovealNet process to be ready
        comm.Barrier()
        # torch.cuda.empty_cache()


        # precompute do not need gaze prediction

        # copy from render()

        # Set up rasterization configuration
        tanfovx = math.tan(view.FoVx * 0.5)
        tanfovy = math.tan(view.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(view.image_height),
            image_width=int(view.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=view.world_view_transform,
            projmatrix=view.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=view.camera_center,
            prefiltered=False,
            debug=pipeline.debug
        )

        colors_precomp = torch.Tensor([])
        cov3D_precomp = torch.Tensor([])
        # Create empty tensors for buffers
        geomBuffer_precomp = torch.Tensor([]).to(torch.uint8)
        binningBuffer_precomp = torch.Tensor([]).to(torch.uint8)
        imageBuffer_precomp = torch.Tensor([]).to(torch.uint8)
        out_color_precomp = torch.Tensor([])


        rawrasterizer = _RasterizeGaussians
        foveaStep = 0
        gaze_x = 0
        gaze_y = 0
        # gaze_r2 = 1e6
        # gaze_r3 = 1e6
        # gaze_r4 = 1e6


        time_elapsed = 0
        # if idx > 0:
        #     torch.cuda.synchronize()
        
        # starters[idx].record()
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
                gaze_x,  # gaze direction x
                gaze_y,  # gaze direction y
                1e6, 1e6, 1e6,  # radii of the foveal level 2,3,4
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
        # ender.record()  
        # torch.cuda.synchronize()
        # time_elapsed += starter.elapsed_time(ender)

        # print(f"[3DGS] view{idx}, Time for precompute: {time_elapsed:.2f} ms")

        
        foveaStep = 1
        gaze_x = 0
        gaze_y = 0
        # gaze_r2 = 1e6
        # gaze_r3 = 1e6
        # gaze_r4 = 1e6


        # time_elapsed = 0
        # starter1.record()
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
                gaze_x,  # gaze direction x
                gaze_y,  # gaze direction y
                1e6, 1e6, 1e6,  # radii of the foveal level 2,3,4
                rendered_image0,
                # radii_precomp,
                # means2D_precomp,
                # conic_opacity_precomp,
                # geom_rgb_precomp,
                # point_list_precomp,
                # ranges_precomp,
                # tile_AMR_levels_last,
                # tile_AMR_levels_current,
                geomBuffer,
                binningBuffer,
                imageBuffer,
                False, # interpolate_image
            raster_settings,
        )
        # ender1.record()  
        # torch.cuda.synchronize()
        # time_elapsed += starter.elapsed_time(ender1)

        # print(f"[3DGS] view{idx}, Time for step1: {time_elapsed:.2f} ms")

        # enders[idx].record()
        # torch.cuda.synchronize()
        # time_elapsed += starter.elapsed_time(ender)
        # print(f"[3DGS] view{idx}, Time for precompute and step1: {time_elapsed:.2f} ms")

        for img_idx in range(50):
            foveaStep = 0
            while foveaStep < 4: # i.e. render until reach highest foveal level
                # sync gaze prediction
                win1.Lock(1)
                local_gaze_buffer = np.array(gaze_predictions_buffer)  # Make a local copy
                print(f"Received gaze prediction: {local_gaze_buffer[img_idx]}")
                win1.Unlock(1)

                win2.Lock(1)
                local_fovealnet_level_buffer = np.array(fovealnet_level_buffer)  # Make a local copy
                print(f"Received fovealnet level: {local_fovealnet_level_buffer[img_idx]}")
                win2.Unlock(1)

                if foveaStep == local_fovealnet_level_buffer[img_idx]:
                    # wait for the other rank 1 process to uupdate the gaze prediction intermediate step
                    time.sleep(0.01)
                    continue

                foveaStep = local_fovealnet_level_buffer[img_idx]
                gaze_x = local_gaze_buffer[img_idx][0]
                gaze_y = local_gaze_buffer[img_idx][1]


                # render according to this foveal level
                # time = 0
                # starter.record()
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
                        gaze_x,  # gaze direction x
                        gaze_y,  # gaze direction y
                        args.gaze_r2,args.gaze_r3,args.gaze_r4,  # radii of the foveal level 2,3,4
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
                # ender.record()  
                # torch.cuda.synchronize()
                # time += starter.elapsed_time(ender)

                # print(f"[3DGS] view{idx}, gaze_idx{img_idx}, Time for compute up to foveal level {foveaStep}: {time:.2f} ms")



else:  # FovealNet process

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    print(f"FovealNet process is using {num_sms} stream multiprocessors.")


    model = VisionTransformer(num_layers=6, top_k=1.0).to(device)
    model.load_state_dict(torch.load(args.foveal_model_path, map_location=device))
    model.eval()

    if args.eye_image_sequence_id_start is not None and args.eye_image_sequence_id_end is not None:
        

        for seq_id in range(args.eye_image_sequence_id_start, args.eye_image_sequence_id_end ):
            # Wait for Gaussian Splatting process to be ready
            comm.Barrier()
            torch.cuda.empty_cache()
            
            # Create CUDA events for timing if foveal_layer_timer is enabled
            if args.foveal_layer_timer:
                num_events = len(model.transformer_layers) + 2  # +1 for patch embedding, +1 for final layers
                starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
                enders = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
            else:
                starters, enders = None, None
            sequence_folder = os.path.join(args.eye_image_sequence_folder, f"{seq_id:04d}")
            if not os.path.exists(sequence_folder):
                print(f"Skipping sequence {seq_id} as it does not exist")
                continue

            # Process images and make predictions
            predictions = []
            total_time = 0
            num_images = 0
            layer_times = [0] * (len(model.transformer_layers) + 2) if args.foveal_layer_timer else None

            local_predictions_buffer = np.zeros((150, 2), dtype=np.float32) 
            local_fovealnet_level_buffer = np.zeros(150, dtype=np.int32)

            # for image_name in tqdm(os.listdir(sequence_folder), desc=f"Processing sequence {seq_id}"):
            # Remove tqdm and process images without progress bar
            img_idx = -1
            for image_name in os.listdir(sequence_folder):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(sequence_folder, image_name)
                    image = load_image(image_path).to(device)
                    img_idx += 1

                    with torch.no_grad():
                        start_time = torch.cuda.Event(enable_timing=True)
                        end_time = torch.cuda.Event(enable_timing=True)
                        start_time.record()
                        
                        if args.foveal_layer_timer:
                            output = model.forward_timer(image, starters, enders)
                        else:
                            output = model(image)
                        
                        end_time.record()
                        # torch.cuda.synchronize()
                        # elapsed_time = start_time.elapsed_time(end_time)
                        # total_time += elapsed_time
                        num_images += 1

                        # if args.foveal_layer_timer:
                        #     for i in range(len(layer_times)):
                        #         layer_times[i] += starters[i].elapsed_time(enders[i])
            
                    prediction = output.cpu().numpy()[0]
                    local_predictions_buffer[img_idx] = prediction  
                    local_fovealnet_level_buffer[img_idx] = 4
                    # # Send gaze prediction to Gaussian Splatting process
                    # comm.Send(gaze_prediction, dest=0)

                    # # Update the shared gaze prediction
                    # win.Lock(0)  # Lock for writing
                    # win.Put(prediction, 0)
                    # print(f"Write prediction to shared memory: {prediction}")
                    # win.Unlock(0)

                    win1.Lock(0)
                    win1.Put(local_predictions_buffer, 0)
                    win1.Unlock(0)

                    win2.Lock(0)
                    win2.Put(local_fovealnet_level_buffer, 0)
                    win2.Unlock(0)
            

                    predictions.append((image_name, prediction))

            # average_time = total_time / num_images
            # print(f"Average inference time for sequence {seq_id}: {average_time:.2f} ms")

            # if args.foveal_layer_timer:
            #     print("\nLayer-wise timing:")
            #     print(f"Patch embedding time: {layer_times[0]/num_images:.2f} ms")
            #     for i, time in enumerate(layer_times[1:-1], 1):
            #         print(f"Transformer block {i} time: {time/num_images:.2f} ms")
            #     print(f"Final layers time: {layer_times[-1]/num_images:.2f} ms")

            # Write predictions to output file
            output_file = f"predictions_{seq_id}.txt"
            with open(output_file, 'w') as f:
                for image_name, prediction in predictions:
                    f.write(f"{image_name}: Pitch={prediction[0]:.4f}, Yaw={prediction[1]:.4f}\n")
    

# Clean up
win1.Free()
win2.Free()