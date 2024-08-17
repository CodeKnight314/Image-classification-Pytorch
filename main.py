import argparse
from model_loader import load_model_class
from dataset import load_dataset
from utils.log_writer import LOGWRITER
from utils.early_stop import EarlyStopMechanism
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
import time
from torch.utils.tensorboard import SummaryWriter

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_model(args, model_config, logger):
    model_class = load_model_class(args.model)
    if args.model == "ViT":
        model = model_class(input_dim=(3, configs.img_height, configs.img_width),
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
    else:
        model = model_class(num_classes=configs.num_class)
        logger.write(f"[INFO] {args.model} Model loaded with defined parameters")

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

def classification(model, optimizer, scheduler, train_dl, valid_dl, logger, writer, loss_fn, epochs, warmup, device='cuda'):    
    es_mech = EarlyStopMechanism(metric_threshold=0.015, mode='min', grace_threshold=10, save_path=configs.save_pth)

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        total_train_loss = 0
        
        # Training Loop
        train_start_time = time.time()
        for images, labels in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            total_train_loss += loss.item()

        train_time = time.time() - train_start_time
        model.eval()
        total_val_loss = 0
        all_labels = []
        all_preds = []

        # Validation Loop
        val_start_time = time.time()
        with torch.no_grad():
            for images, labels in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_time = time.time() - val_start_time

        # Timing the code block for computing class prediction stats
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        stats_start_time = time.time()

        pred_counter = Counter(all_preds)
        num_classes = len(valid_dl.dataset.id_to_class_dict)

        no_preds_classes = [cls for cls in range(num_classes) if pred_counter[cls] == 0]
        if no_preds_classes:
            no_preds_class_names = [valid_dl.dataset.id_to_class_dict[cls] for cls in no_preds_classes]
            logger.write(f"[INFO] Classes with no predicted samples: {len(no_preds_class_names)}")

        stats_time = time.time() - stats_start_time

        # Timing the code block for computing precision, recall, accuracy
        metrics_start_time = time.time()

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)

        avg_train_loss = total_train_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(valid_dl)

        metrics_time = time.time() - metrics_start_time

        # Timing the code block for early stopping mechanism
        early_stop_start_time = time.time()

        es_mech.step(model=model, metric=avg_val_loss)    
        if es_mech.check(): 
            logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
            break

        early_stop_time = time.time() - early_stop_start_time

        logger.log_results(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss, precision=precision, recall=recall, accuracy=accuracy)

        # Logging metrics and times to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)
        writer.add_scalar('Metrics/Precision', precision, epoch+1)
        writer.add_scalar('Metrics/Recall', recall, epoch+1)
        writer.add_scalar('Metrics/Accuracy', accuracy, epoch+1)
        writer.add_scalar('Time/Train', train_time, epoch+1)
        writer.add_scalar('Time/Validation', val_time, epoch+1)
        writer.add_scalar('Time/Stats Computation', stats_time, epoch+1)
        writer.add_scalar('Time/Metrics Computation', metrics_time, epoch+1)
        writer.add_scalar('Time/Early Stopping Check', early_stop_time, epoch+1)

        # Logging pie chart for the computation times (as scalars for each component)
        total_time = train_time + val_time + stats_time + metrics_time + early_stop_time
        writer.add_scalars('Computation Time Distribution', {
            'Training': train_time / total_time,
            'Validation': val_time / total_time,
            'Stats Computation': stats_time / total_time,
            'Metrics Computation': metrics_time / total_time,
            'Early Stopping Check': early_stop_time / total_time
        }, epoch+1)

        if epoch > warmup:
            scheduler.step()

        epoch_time = time.time() - start_time
        logger.write(f"[INFO] Epoch {epoch+1} completed in {epoch_time:.2f} seconds (Training: {train_time:.2f}s, Validation: {val_time:.2f}s, Stats: {stats_time:.2f}s, Metrics: {metrics_time:.2f}s, Early Stopping: {early_stop_time:.2f}s).")

def main():
    parser = argparse.ArgumentParser(description='Train a model on Image Classification')
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 
                                                                     'ResNet18', 
                                                                     'ResNet34',
                                                                     'CvT-13',
                                                                     'CvT-21',
                                                                     'CvT-24',
                                                                     'MobileNet',
                                                                     'Squeezenetv3',
                                                                     'InceptionNetv3',
                                                                     'VGG16',
                                                                     'VGG19',
                                                                     'DenseNet121',
                                                                     'DenseNet169',
                                                                     'DenseNet201',
                                                                     'DenseNet264',
                                                                     'EfficientNetV2'], help='Model name')
    
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--config_file', type=str, required=True, default='config.json', help='Path to configuration file')

    args = parser.parse_args()
    
    model_config = load_config(args.config_file)
    configs.trial_directory()

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=model_config.get('epochs'))
    logger.write(f"[INFO] Log writer loaded and binded to {configs.log_output_dir}")
    logger.write(f"[INFO] Total epochs: {model_config.get('epochs')}")
    logger.write(f"[INFO] Warm Up Phase: {model_config.get('warm_up_epochs')} epochs")

    # TensorBoard Writer Initialization
    writer = SummaryWriter(log_dir=configs.log_output_dir)
    logger.write("[INFO] TensorBoard writer initialized.")

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

    classification(model=model, 
                       optimizer=optimizer, 
                       scheduler=scheduler, 
                       train_dl=train_dl, 
                       valid_dl=valid_dl, 
                       logger=logger, 
                       writer=writer, 
                       loss_fn=loss_fn, 
                       epochs=model_config.get("epochs"),
                       warmup=model_config.get("warmup"))

    writer.close()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
