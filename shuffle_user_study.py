from PIL import Image
import os
import shutil
import random

# Define the source directories
foveated_marked_dir = "user_study/foveated_marked"
original_marked_dir = "user_study/original_marked"

# Define the destination directory
destination_dir = "user_study_images"
os.makedirs(destination_dir, exist_ok=True)
os.makedirs(destination_dir+"/original", exist_ok=True)
os.makedirs(destination_dir+"/foveated", exist_ok=True)

# Collect all image names from one of the directories
image_names = [f for f in os.listdir(foveated_marked_dir) if f.endswith('.png')]

# Shuffle the image names
random.shuffle(image_names)

# Copy, crop, and rename images
for i, image_name in enumerate(image_names, start=1):
    # Define source paths
    foveated_marked_path = os.path.join(foveated_marked_dir, image_name)
    original_marked_path = os.path.join(original_marked_dir, image_name)
    
    # Open, crop, and save the foveated image
    with Image.open(foveated_marked_path) as img:
        cropped_img = img.crop((0, 0, img.width - 1, img.height - 1))
        foveated_marked_dest = os.path.join(destination_dir, f"foveated/{i}.png")
        cropped_img.save(foveated_marked_dest)
    
    # Open, crop, and save the original image
    with Image.open(original_marked_path) as img:
        cropped_img = img.crop((0, 0, img.width - 1, img.height - 1))
        original_marked_dest = os.path.join(destination_dir, f"original/{i}.png")
        cropped_img.save(original_marked_dest)

print("Images have been shuffled, cropped, and copied to", destination_dir)

# then pair them

# Define the source directory
source_dir = "user_study_images"

# Define the destination directory
pairs_dir = "user_study_pairs"
os.makedirs(pairs_dir, exist_ok=True)

# Prepare the answer file
answer_file_path = os.path.join(pairs_dir, "answer.txt")
answer_file = open(answer_file_path, "w")

# Process each image pair
for i in range(1, 21):
    # Create a directory for each pair
    pair_dir = os.path.join(pairs_dir, str(i))
    os.makedirs(pair_dir, exist_ok=True)
    
    # Define the paths for the foveated and original images
    foveated_image = os.path.join(source_dir, f"foveated/{i}.png")
    original_image = os.path.join(source_dir, f"original/{i}.png")
    
    # Randomly assign 'a' or 'b' to the images
    if random.choice([True, False]):
        a_image, b_image = foveated_image, original_image
        answer = "a"
    else:
        a_image, b_image = original_image, foveated_image
        answer = "b"
    
    # Copy the images to the pair directory
    shutil.copy(a_image, os.path.join(pair_dir, "a.png"))
    shutil.copy(b_image, os.path.join(pair_dir, "b.png"))
    
    # Record the answer
    answer_file.write(f"{i}: {answer}\n")

# Close the answer file
answer_file.close()

print("Image pairs have been created and recorded in", pairs_dir)