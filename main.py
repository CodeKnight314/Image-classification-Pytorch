import argparse
from models import ResNet, ViT, CvT, MobileNet, Squeezenet
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

def load_model(args, model_config, logger):
    # Model selection
    if args.model == "ViT":
        model = ViT.get_ViT(input_dim=(3, configs.img_height, configs.img_width),
                        patch_size=model_config.get("patch_size"), 
                        layers=model_config.get("layers"), 
                        d_model=model_config.get("d_model"), 
                        head=model_config.get("head"), 
                        num_classes=configs.num_class)
        logger.write("[INFO] ViT Model loaded with the following attributes:")
        logger.write(f"[INFO] * Patch size: {model.patch_size}.")
        logger.write(f"[INFO] * Number of layers: {model.layers}.")
        logger.write(f"[INFO] * Model dimension: {model.d_model}.")
        logger.write(f"[INFO] * Number of attention heads: {model.head}.")
    elif args.model == "ResNet18":
        model = ResNet.get_ResNet18(num_classes=configs.num_class)
        logger.write("[INFO] ResNet18 Model loaded with defined parameters")
    elif args.model == "ResNet34":
        model = ResNet.get_ResNet34(num_classes=configs.num_class)
        logger.write("[INFO] ResNet34 Model loaded with defined parameters")
    elif args.model == "CvT-13": 
        model = CvT.get_CVT13(num_classes=configs.num_class)
        logger.write("[INFO] CvT-13 Model loaded with defined parameters")
    elif args.model == "CvT-21": 
        model = CvT.get_CVT21(num_classes=configs.num_class)
        logger.write("[INFO] CvT-21 Model loaded with defined parameters")
    elif args.model == "CvT-24":
        model = CvT.get_CVTW24(num_classes=configs.num_class)
        logger.write("[INFO] CvT-24 Model loaded with defined parameters")
    elif args.model == "MobileNet":
        model = MobileNet.get_MobileNet(num_of_classes=configs.num_class)
        logger.write("[INFO] MobileNet loaded with defined parameters.")
    elif args.model == "Squeezenetv1": 
        model = Squeezenet.get_SqueezenetV1(num_classes=configs.num_class)
        logger.write("[INFO] Squeezenetv1 loaded with defined parameters")
    elif args.model == "Squeezenetv2": 
        model = Squeezenet.get_SqueezenetV2(num_classes=configs.num_class)
        logger.write("[INFO] Squeezenetv2 loaded with defined parameters")
    elif args.model == "Squeezenetv3": 
        model = Squeezenet.get_SqueezenetV3(num_classes=configs.num_class)
        logger.write("[INFO] Squeezenetv3 loaded with defined parameters")

    # Weights loading
    if args.model_save_path:
        logger.write("[INFO] Model weights provided. Attempting to load model weights.")
        try:
            model.load_state_dict(torch.load(args.model_save_path), strict=False)
            logger.log_error("[INFO] Model weights loaded successfully with strict=False.")
        except RuntimeError as e:
            logger.log_error(f"[WARNING] Runtime error occurred while loading some model weights: {e}")
        except FileNotFoundError as e:
            logger.log_error(f"[ERROR] File not found error occurred: {e}")
        except Exception as e:
            logger.log_error(f"[ERROR] An unexpected error occurred while loading model weights: {e}")
    else:
        logger.write("[INFO] No model weights path provided. Training from scratch.")

    return model

def classification(model, optimizer, scheduler, train_dl, valid_dl, logger, loss_fn, epochs, warmup, device='cuda'):
    best_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):

        model.train()
        total_train_loss = 0

        # Training loop
        for images, labels in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        all_labels = []
        all_preds = []
        
        # Validation Loop
        with torch.no_grad():
            for images, labels in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # Data and variable preparations
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        pred_counter = Counter(all_preds)
        
        num_classes = len(valid_dl.dataset.id_to_class_dict)
        
        # Checking and reporting if model is not classifiying missing classes
        no_preds_classes = [cls for cls in range(num_classes) if pred_counter[cls] == 0]
        if no_preds_classes:
            no_preds_class_names = [valid_dl.dataset.id_to_class_dict[cls] for cls in no_preds_classes]
            logger.write(f"[INFO] Classes with no predicted samples: {len(no_preds_class_names)}")
        
        # Validation metrics
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        
        avg_train_loss = total_train_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(valid_dl)

        # Saving best model based on avg validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(configs.save_pth, f'Best_model_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

        # logging results to linked directory
        logger.log_results(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss,
                     precision=precision, recall=recall, accuracy=accuracy)

        # Scheduler Stepping once warmup phase ends
        if epoch > warmup:
            scheduler.step()

def main():
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 
                                                                     'ResNet18', 
                                                                     'ResNet34',
                                                                     'CvT-13',
                                                                     'CvT-21',
                                                                     'CvT-24',
                                                                     'MobileNet',
                                                                     'Squeezenetv1',
                                                                     'Squeezenetv2',
                                                                     'Squeezenetv3'], help='Model name')
    
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--config_file', type=str, required=True, default='config.json', help='Path to configuration file')

    args = parser.parse_args()
    
    model_config = load_config(args.config_file)

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=model_config.get('epochs'))
    logger.write(f"[INFO] Log writer loaded and binded to {configs.log_output_dir}")
    logger.write(f"[INFO] Total epochs: {model_config.get('epochs')}")
    logger.write(f"[INFO] Warm Up Phase: {model_config.get('warm_up_epochs')} epochs")

    train_dl = load_dataset(root_dir=args.root_dir, mode="train")
    valid_dl = load_dataset(root_dir=args.root_dir, mode="val")
    logger.write(f"[INFO] Training Dataloader loaded with {len(train_dl)} batches.")
    logger.write(f"[INFO] Validation Dataloader loaded with {len(valid_dl)} batches.")
    logger.write(f"[INFO] Total number of classes: {configs.num_class}")

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
    logger.write("[INFO] Cross Entropy Function loaded.")

    model = load_model(args, model_config, logger)
                
    if model_config.get("optimizer") == 'AdamW':
        optimizer = opt.AdamW(model.parameters(), lr=model_config.get("lr"), weight_decay=model_config.get("weight decay"))
    elif model_config.get("optimizer") == 'SGD':
        optimizer = opt.SGD(model.parameters(), lr=model_config.get("lr"), weight_decay=model_config.get("weight decay"), momentum=0.9)
    logger.write(f"[INFO] Optimizer loaded with learning rate: {model_config.get('lr')}.")

    if model_config.get("scheduler") == 'CosineAnnealingLR':
        scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config.get("t_max"), eta_min=model_config.get("eta_min"))
    elif model_config.get("scheduler") == 'StepLR':
        scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=model_config.get("step_size"), gamma=model_config.get("gamma"))
    logger.write(f"[INFO] {model_config.get('scheduler')} Scheduler loaded.")

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