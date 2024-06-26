import argparse
from models.ViT import get_ViT
from models.ResNet import get_ResNet18, get_ResNet34
from dataset import load_dataset
from utils.log_writer import LOGWRITER
import torch
import torch.optim as opt
from tqdm import tqdm
import os
import configs
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch.multiprocessing as mp
import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def classification(model, optimizer, scheduler, train_dl, valid_dl, logger, loss_fn, epochs, warmup, device='cuda'):
    best_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        pred_counter = Counter(all_preds)
        
        num_classes = len(valid_dl.dataset.id_to_class_dict)
        
        no_preds_classes = [cls for cls in range(num_classes) if pred_counter[cls] == 0]
        if no_preds_classes:
            no_preds_class_names = [valid_dl.dataset.id_to_class_dict[cls] for cls in no_preds_classes]
            print(f"[INFO] Classes with no predicted samples: {len(no_preds_class_names)}")
        
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        
        avg_train_loss = total_train_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(valid_dl)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(configs.save_pth, f'Best_model_CIFAR10_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

        logger.write(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss,
                     precision=precision, recall=recall, accuracy=accuracy)

        if epoch > warmup:
            scheduler.step()

def main():
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 'ResNet18', 'ResNet34'], help='Model name')
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--config_file', type=str, required=True, default='config.json', help='Path to configuration file')

    args = parser.parse_args()
    
    model_config = load_config(args.config_file)

    train_dl = load_dataset(root_dir=args.root_dir, mode="train")
    valid_dl = load_dataset(root_dir=args.root_dir, mode="val")
    print(f"[INFO] Training Dataloader loaded with {len(train_dl)} batches.")
    print(f"[INFO] Validation Dataloader loaded with {len(valid_dl)} batches.")
    print(f"[INFO] Total number of classes: {configs.num_class}")

    loss_fn = torch.nn.CrossEntropyLoss()
    print("[INFO] Cross Entropy Function loaded.")

    if args.model == "ViT":
        model = get_ViT(input_dim=(3, configs.img_height, configs.img_width),
                        patch_size=model_config.get("patch_size"), 
                        layers=model_config.get("layers"), 
                        d_model=model_config.get("d_model"), 
                        head=model_config.get("head"), 
                        num_classes=configs.num_class)
        print("[INFO] ViT Model loaded with the following attributes:")
        print(f"[INFO] * Patch size: {model.patch_size}.")
        print(f"[INFO] * Number of layers: {model.layers}.")
        print(f"[INFO] * Model dimension: {model.d_model}.")
        print(f"[INFO] * Number of attention heads: {model.head}.")
    elif args.model == "ResNet18":
        model = get_ResNet18(num_classes=configs.num_class)
        print("[INFO] ResNet18 Model loaded with the following attributes:")
        print(f"[INFO] * Channels: {model.channels}")
        print(f"[INFO] * Layers: {model.num_layers}")
    elif args.model == "ResNet34":
        model = get_ResNet34(num_classes=configs.num_class)
        print("[INFO] ResNet34 Model loaded with the following attributes:")
        print(f"[INFO] * Channels: {model.channels}")
        print(f"[INFO] * Layers: {model.num_layers}")

    if args.model_save_path:
        print("[INFO] Model weights provided. Loading model weights.")
        model.load_state_dict(torch.load(args.model_save_path))
    
    if model_config.get("optimizer") == 'AdamW':
        optimizer = opt.AdamW(model.parameters(), lr=model_config.get("lr"), weight_decay=model_config.get("weight decay"))
    elif model_config.get("optimizer") == 'SGD':
        optimizer = opt.SGD(model.parameters(), lr=model_config.get("lr"), weight_decay=model_config.get("weight decay"), momentum=0.9)
    print(f"[INFO] Optimizer loaded with learning rate: {model_config.get('lr')}.")

    if model_config.get("scheduler") == 'CosineAnnealingLR':
        scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config.get("t_max"), eta_min=model_config.get("eta_min"))
    elif model_config.get("scheduler") == 'StepLR':
        scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=model_config.get("step_size"), gamma=model_config.get("gamma"))
    print(f"[INFO] {model_config.get('scheduler')} Scheduler loaded.")

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=model_config.get('epochs'))
    print(f"[INFO] Log writer loaded and binded to {configs.log_output_dir}")
    print(f"[INFO] Total epochs: {model_config.get('epochs')}")
    print(f"[INFO] Warm Up Phase: {model_config.get('warm_up_epochs')} epochs")

    configs.trial_directory()

    classification(model=model, 
                       optimizer=optimizer, 
                       scheduler=scheduler, 
                       train_dl=train_dl, 
                       valid_dl=valid_dl, 
                       logger=logger, 
                       loss_fn=loss_fn, 
                       epochs=model_config.get("epochs"),
                       warmup=model_config.get("warmup"))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()