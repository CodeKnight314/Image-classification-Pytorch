import argparse  
import torch 
import torch.nn as nn 
import os 
from tqdm import tqdm
from utils.visualization import * 
from torch.utils.data import DataLoader 
from loss import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc
from dataset import load_dataset
from models.ResNet import get_ResNet18, get_ResNet34
from models.ViT import get_ViT
import configs

device = configs.device

def binarize_labels(labels, num_classes): 
    """
    Assumes labels is 1D
    """
    matrix = np.zeros((labels.shape[0], num_classes))
    for i, label in enumerate(labels):  # Fix variable name
        matrix[i][label] = 1  # Fix indexing
    return matrix

def compute_tpr_fpr(labels_sorted, scores_sorted): 
    """
    Compute TPR and FPR given sorted labels and scores.
    """
    tpr = [] 
    fpr = [] 
    pos_count = np.sum(labels_sorted)
    neg_count = len(labels_sorted) - pos_count

    tp = 0 
    fp = 0 

    for threshold in np.unique(scores_sorted):
        tp = np.sum((scores_sorted >= threshold) & (labels_sorted == 1))
        fp = np.sum((scores_sorted >= threshold) & (labels_sorted == 0))
        
        tpr.append(tp / pos_count)
        fpr.append(fp / neg_count)
    
    return np.array(tpr), np.array(fpr)

def compute_auc(fpr, tpr):
    """
    Compute AUC using the trapezoidal rule.
    """
    return np.trapz(tpr, fpr)

def ROC_AUC_Curve_plot(labels, logits, num_classes, output_dir):
    """
    Plot ROC curves and compute AUC for each class.
    """
    binary_labels = binarize_labels(labels=labels, num_classes=num_classes)
    
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes): 
        scores_sorted = np.sort(logits[:, i])[::-1]
        sorted_indices = np.argsort(logits[:, i])[::-1]
        labels_sorted = binary_labels[sorted_indices, i]

        fpr[i], tpr[i] = compute_tpr_fpr(labels_sorted=labels_sorted, scores_sorted=scores_sorted)
        roc_auc[i] = compute_auc(fpr=fpr[i], tpr=tpr[i])

        plt.figure(figsize=(10, 8))
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - Class {i}')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join(output_dir, f'roc_curve_class_{i}.png'))
        plt.close()

def evaluation(model : nn.Module, valid_dl : DataLoader, output_directory : str):
    """
    Evaluation of model on image classification. 

    Args: 
        model (nn.Module): Model to evaluate on image classification. 
        valid_dl (DataLoader): Dataloader of the validation or test dataset for evaluation. 
        output_directory (str): Path to output heatmaps and confusion matrices.
    """
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_val_loss = 0
    all_labels = []
    all_preds = []
    all_logits = []
    
    # Validation Loop
    with torch.no_grad():
        for images, labels in tqdm(valid_dl, desc=f"[Model Evaluation]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
            outputs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    
    pred_counter = Counter(all_preds)

    num_classes = len(valid_dl.dataset.id_to_class_dict)

    # ROC & AUC
    ROC_AUC_Curve_plot(all_labels, all_logits, num_classes=num_classes, output_dir=output_directory)

    # Confusion Matrix
    if(num_classes > 10):
        print("[INFO] Total number of classes exceeds 10 and would result in a large confusion matrix.")
        usr_input = input(f"[QUERY] Do you wish to continue with constructing the {num_classes} x {num_classes} confusion matrix? [Y/n]")
        if(usr_input == "Y"):
            labels = torch.from_numpy(all_labels)
            predictions = torch.from_numpy(all_preds)

            cm = confusion_matrix(labels=labels, predictions=predictions, num_class=num_classes)

            plot_confusion_matrix(confusion_matrix=cm, num_classes=num_classes, save_pth=os.path.join(output_directory, "Confusion Matrix.png"))
            print("[INFO] Confusion Matrix saved to: {}".format(output_directory))
        else: 
            print("[INFO] Ignoring confusion matrix construction.")
    else: 
        labels = torch.from_numpy(all_labels)
        predictions = torch.from_numpy(all_preds)

        cm = confusion_matrix(labels=labels, predictions=predictions, num_class=num_classes)

        plot_confusion_matrix(confusion_matrix=cm, num_classes=num_classes, save_pth=os.path.join(output_directory, "Confusion Matrix.png"))
        print("[INFO] Confusion Matrix saved to: {}".format(output_directory))
    
    # Missing Classes
    no_preds_classes = [cls for cls in range(num_classes) if pred_counter[cls] == 0]
    if no_preds_classes:
        no_preds_class_names = [valid_dl.dataset.id_to_class_dict[cls] for cls in no_preds_classes]
        print(f"[INFO] Classes with no predicted samples: {no_preds_class_names}")
    
    # Classification Metrics
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    avg_val_loss = total_val_loss / len(valid_dl)

    print(f"[INFO] Precision over the entire dataset: {precision}")
    print(f"[INFO] Accuracy over the entire dataset: {accuracy}")
    print(f"[INFO] Recall over the entire dataset: {recall}")
    print(f"[INFO] Average Loss over the entire dataset: {avg_val_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 'ResNet18', 'ResNet34'], help='Model name')
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights.')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output confusion matrix and attention maps.")

    args = parser.parse_args()

    valid_dl = load_dataset(root_dir=args.root_dir, mode="val") 
    print("[INFO] Evaluation dataset constructed.")
    if args.model == "ViT":
        model = get_ViT(patch_size=args.patch_size, num_classes=configs.num_class)
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

    
