import torch 
import os

def count_folders_with_prefix(directory, prefix):
    count = 0
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith(prefix):
                count += 1
    return count

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = None

img_height = 224
img_width = 224

transforms = None

batch_size = 128

base_path = os.getcwd()
prefix = "Trial"
output_dir = os.path.join(base_path, f'{prefix}_{count_folders_with_prefix(base_path, prefix)+1}')

log_output_dir = os.path.join(output_dir, "log_outputs")

save_pth = os.path.join(output_dir, "saved_weights")

num_class = None

id_to_category_dict = None

category_to_id_dict = None

def trial_directory(): 
    if not os.path.exists(log_output_dir): 
        os.makedirs(log_output_dir)
        print("[INFO] Creating Log output directory.")

    if not os.path.exists(save_pth): 
        os.makedirs(save_pth)
        print("[INFO] Creating save path directory.")