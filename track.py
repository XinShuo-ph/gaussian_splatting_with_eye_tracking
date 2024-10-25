import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
from fovealnet.timm_vit import VisionTransformer
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for gaze estimation model.")
    parser.add_argument("--foveal_model_path", default="/home/ubuntu/gaussian-splatting/fovealnet/results/model_minmax_0.8.pt", type=str, help="Path to the trained model")
    parser.add_argument("--eye_image_folder", default="/home/ubuntu/openeds/test/sequences/0000/", type=str, help="Folder containing input images")
    parser.add_argument("--eye_image_sequence_folder", default="/home/ubuntu/openeds/test/sequences/", type=str, help="Folder containing input image sequences")
    parser.add_argument("--eye_image_sequence_id_start", default=None, type=int, help="Start of the image sequence ID range")
    parser.add_argument("--eye_image_sequence_id_end", default=None, type=int, help="End of the image sequence ID range")
    parser.add_argument("--foveal_output_file", type=str, default="predictions.txt", help="Output file for predictions")
    parser.add_argument("--foveal_layer_timer", action="store_true", help="Enable layer-wise timing")
    return parser.parse_args()

def load_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = VisionTransformer(num_layers=6, top_k=1.0).to(device)
model.load_state_dict(torch.load(args.foveal_model_path, map_location=device))
model.eval()


if args.eye_image_sequence_id_start is not None and args.eye_image_sequence_id_end is not None:
    

    for seq_id in range(args.eye_image_sequence_id_start, args.eye_image_sequence_id_end + 1):
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

        for image_name in tqdm(os.listdir(sequence_folder), desc=f"Processing sequence {seq_id}"):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(sequence_folder, image_name)
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
        
                gaze_prediction = output.cpu().numpy()[0]
                predictions.append((image_name, gaze_prediction))

        average_time = total_time / num_images
        print(f"Average inference time for sequence {seq_id}: {average_time:.2f} ms")

        if args.foveal_layer_timer:
            print("\nLayer-wise timing:")
            print(f"Patch embedding time: {layer_times[0]/num_images:.2f} ms")
            for i, time in enumerate(layer_times[1:-1], 1):
                print(f"Transformer block {i} time: {time/num_images:.2f} ms")
            print(f"Final layers time: {layer_times[-1]/num_images:.2f} ms")

        # Write predictions to output file
        # output_file = f"predictions_{seq_id}.txt"
        # with open(output_file, 'w') as f:
        #     for image_name, prediction in predictions:
        #         f.write(f"{image_name}: Pitch={prediction[0]:.4f}, Yaw={prediction[1]:.4f}\n")
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
    
            gaze_prediction = output.cpu().numpy()[0]
            predictions.append((image_name, gaze_prediction))

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
