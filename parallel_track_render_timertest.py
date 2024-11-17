from mpi4py import MPI
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_amr import render, GaussianModel
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
parser.add_argument("--eye_image_sequence_id_start", default=None, type=int)
parser.add_argument("--eye_image_sequence_id_end", default=None, type=int)
parser.add_argument("--foveal_output_file", type=str, default="predictions.txt")
parser.add_argument("--foveal_layer_timer", action="store_true")

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--show_fps", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
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
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # record 4 fov steps separately
    starter0, ender0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter3, ender3 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter4, ender4 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    # choose to record time or fps
    if args.show_fps:
        fpss = []
        fpss0 = []
        fpss1 = []
        fpss2 = []
        fpss3 = []
        fpss4 = []
    else:
        times = []
        times0 = []
        times1 = []
        times2 = []
        times3 = []
        times4 = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # Wait for FovealNet process to be ready
        comm.Barrier()
        
        # Use gaze prediction to adjust rendering if needed
        # For now, we'll just print it
        # print(f"Received gaze prediction: Pitch={gaze_prediction[0]:.4f}, Yaw={gaze_prediction[1]:.4f}")

        # instead of sending and receiving, use windows to share gaze prediction
        # Read from shared memory
        # win.Lock(1)  # Lock for reading
        # print(f"Received gaze prediction: {sync_gaze_prediction}")
        # local_gaze = np.array(sync_gaze_prediction)  # Make a local copy
        # print(f"Received gaze prediction: {local_gaze}")
        # win.Unlock(1)




        time = 0
        time0 = 0
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        for i in range(5):
            rendering = render(view, gaussians, pipeline, background,starter = starter, ender= ender, 
                               starters = [starter0, starter1, starter2, starter3, starter4], enders = [ender0, ender1, ender2, ender3, ender4],
                               test_no_render_laststep=args.test_no_render_laststep
                               )["render"]
            
            win1.Lock(1)
            local_gaze_buffer = np.array(gaze_predictions_buffer)  # Make a local copy
            # print(f"Received gaze prediction: {local_gaze_buffer}")
            print(f"Received gaze prediction: {local_gaze_buffer[local_gaze_buffer[:,0] != 0]}")
            win1.Unlock(1)

            win2.Lock(1)
            local_fovealnet_level_buffer = np.array(fovealnet_level_buffer)  # Make a local copy
            # print(f"Received fovealnet level: {local_fovealnet_level_buffer}")
            print(f"Received fovealnet level: {local_fovealnet_level_buffer[local_fovealnet_level_buffer != 0]}")
            win2.Unlock(1)
            torch.cuda.synchronize()
            time += starter.elapsed_time(ender)
            time0 += starter0.elapsed_time(ender0)
            time1 += starter1.elapsed_time(ender1)
            time2 += starter2.elapsed_time(ender2)
            time3 += starter3.elapsed_time(ender3)
            time4 += starter4.elapsed_time(ender4)
        if args.show_fps:
            # count fps every 5 frames
            fps = 5 / (time / 1000)
            fps0 = 5 / (time0 / 1000)
            fps1 = 5 / (time1 / 1000)
            fps2 = 5 / (time2 / 1000)
            fps3 = 5 / (time3 / 1000)
            fps4 = 5 / (time4 / 1000)
            # print("FPS: ", fps)
            fpss.append(fps)
            fpss0.append(fps0)
            fpss1.append(fps1)
            fpss2.append(fps2)
            fpss3.append(fps3)
            fpss4.append(fps4)
        else:
            times.append(time/5)
            times0.append(time0/5)
            times1.append(time1/5)
            times2.append(time2/5)
            times3.append(time3/5)
            times4.append(time4/5)

    if args.show_fps:      
        avg_fps = sum(fpss) / len(fpss)
        avg_fps0 = sum(fpss0) / len(fpss0)
        avg_fps1 = sum(fpss1) / len(fpss1)
        avg_fps2 = sum(fpss2) / len(fpss2)
        avg_fps3 = sum(fpss3) / len(fpss3)
        avg_fps4 = sum(fpss4) / len(fpss4)

        print(f"Average FPS: {avg_fps}")
        print(f"Average FPS of fov level 0: {avg_fps0}")
        print(f"Average FPS of fov level 1: {avg_fps1}")
        print(f"Average FPS of fov level 2: {avg_fps2}")
        print(f"Average FPS of fov level 3: {avg_fps3}")
        print(f"Average FPS of fov level 4: {avg_fps4}")
    else:
        avg_time = sum(times) / len(times)
        avg_time0 = sum(times0) / len(times0)
        avg_time1 = sum(times1) / len(times1)
        avg_time2 = sum(times2) / len(times2)
        avg_time3 = sum(times3) / len(times3)
        avg_time4 = sum(times4) / len(times4)

        print(f"Average time: {avg_time} ms")
        print(f"Average time of fov level 0: {avg_time0} ms")
        print(f"Average time of fov level 1: {avg_time1} ms")
        print(f"Average time of fov level 2: {avg_time2} ms")
        print(f"Average time of fov level 3: {avg_time3} ms")
        print(f"Average time of fov level 4: {avg_time4} ms")



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
                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(end_time)
                        total_time += elapsed_time
                        num_images += 1

                        if args.foveal_layer_timer:
                            for i in range(len(layer_times)):
                                layer_times[i] += starters[i].elapsed_time(enders[i])
            
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

            average_time = total_time / num_images
            print(f"Average inference time for sequence {seq_id}: {average_time:.2f} ms")

            if args.foveal_layer_timer:
                print("\nLayer-wise timing:")
                print(f"Patch embedding time: {layer_times[0]/num_images:.2f} ms")
                for i, time in enumerate(layer_times[1:-1], 1):
                    print(f"Transformer block {i} time: {time/num_images:.2f} ms")
                print(f"Final layers time: {layer_times[-1]/num_images:.2f} ms")

            # Write predictions to output file
            output_file = f"predictions_{seq_id}.txt"
            with open(output_file, 'w') as f:
                for image_name, prediction in predictions:
                    f.write(f"{image_name}: Pitch={prediction[0]:.4f}, Yaw={prediction[1]:.4f}\n")
    else:
        # Create CUDA events for timing if foveal_layer_timer is enabled
        if args.foveal_layer_timer:
            num_events = len(model.transformer_layers) + 2  # +1 for patch embedding, +1 for final layers
            starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
            enders = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
        else:
            starters, enders = None, None
        # Process images and make predictions
        predictions = []
        total_time = 0
        num_images = 0
        layer_times = [0] * (len(model.transformer_layers) + 2) if args.foveal_layer_timer else None
        
        for image_name in tqdm(os.listdir(args.eye_image_folder), desc="Processing images"):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.eye_image_folder, image_name)
                image = load_image(image_path).to(device)

                with torch.no_grad():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    
                    if args.foveal_layer_timer:
                        output = model.forward_timer(image, starters, enders)
                    else:
                        output = model(image)
                    
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)
                    total_time += elapsed_time
                    num_images += 1

                    if args.foveal_layer_timer:
                        for i in range(len(layer_times)):
                            layer_times[i] += starters[i].elapsed_time(enders[i])
        

                prediction = output.cpu().numpy()[0]
                # # Send gaze prediction to Gaussian Splatting process
                # comm.Send(gaze_prediction, dest=0)

                # # Update the shared gaze prediction
                win.Lock(0)  # Lock for writing
                # sync_gaze_prediction[:] = prediction  # Update shared memory
                win.Put(prediction, 0)
                # print(f"Write prediction to shared memory: {prediction}")
                win.Unlock(0)



                predictions.append((image_name, prediction))

        average_time = total_time / num_images
        print(f"Average inference time: {average_time:.2f} ms")

        if args.foveal_layer_timer:
            print("\nLayer-wise timing:")
            print(f"Patch embedding time: {layer_times[0]/num_images:.2f} ms")
            for i, time in enumerate(layer_times[1:-1], 1):
                print(f"Transformer block {i} time: {time/num_images:.2f} ms")
            print(f"Final layers time: {layer_times[-1]/num_images:.2f} ms")

        # Write predictions to output file
        with open(args.foveal_output_file, 'w') as f:
            for image_name, prediction in predictions:
                f.write(f"{image_name}: Pitch={prediction[0]:.4f}, Yaw={prediction[1]:.4f}\n")

        print(f"Predictions saved to {args.foveal_output_file}")


# Clean up
win.Free()