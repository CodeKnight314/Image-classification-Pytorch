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

root_dir = r"C:\Users\richa\Desktop\cifar10\cifar10"

img_height = 32
img_width = 32

transforms = None

model_save_path = None

batch_size = 128

warm_up_epochs = 10
epochs = 100

ViT_patches = 4
ViT_layers = 8
ViT_d_model = 512 
ViT_head = 4

lr = 1e-4

weight_decay = 1e-4

output_dir = os.path.join(r"C:\Users\richa\Desktop", f"Trial_{count_folders_with_prefix(r"C:\Users\richa\Desktop", "Trial")+1}")

log_output_dir = os.path.join(output_dir, "log_outputs")

matrix_output_dir = os.path.join(output_dir, "conf_matrices")

save_pth = os.path.join(output_dir, "saved_weights")

heatmaps = os.path.join(output_dir, "heatmaps")

num_class = None

id_to_category_dict = None

category_to_id_dict = None

#Available Options: ViT, ResNet18, ResNet34
model = "ResNet18"

def main(): 
    if not os.path.exists(log_output_dir): 
        os.makedirs(log_output_dir)
        print("[INFO] Log output dir not found, Creating folder directory.")

    if not os.path.exists(matrix_output_dir): 
        os.makedirs(matrix_output_dir)
        print("[INFO] Matrix output dir not found. Creating folder directory.")

    if not os.path.exists(save_pth): 
        os.makedirs(save_pth)
        print("[INFO] Model Save path not found. Creating folder directory.")
    
    if not os.path.exists(heatmaps):
        os.makedirs(heatmaps)
        print("[INFO] Heat map directory not found. Creating folder directory.")

if __name__ == "__main__": 
    main()
    