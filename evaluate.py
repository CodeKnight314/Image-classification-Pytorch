import argparse  
import torch 
import torch.nn as nn 
import os 
from tqdm import tqdm
from utils.visualization import * 
from torch.utils.data import DataLoader 
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from dataset import load_dataset
from models import ResNet, ViT, HybridCNNTransformer, MobileNet
import configs
import json

device = configs.device

def load_config(config_file: str) -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def ROC_AUC_Curve_plot(labels: np.ndarray, logits: np.ndarray, num_classes: int, output_dir: str):
    """
    Plot ROC curves and compute AUC for each class using scikit-learn.
    
    Args:
        labels (np.ndarray): True labels, shape (n_samples,).
        logits (np.ndarray): Predicted logits, shape (n_samples, num_classes).
        num_classes (int): Number of classes.
        output_dir (str): Directory to save the ROC curve plots.
    """
    binary_labels = label_binarize(labels, classes=range(num_classes))
    
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

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

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """
    Calculate precision, recall, and F1 score for each class and save the metrics to a file.
    
    Args:
        y_true (np.ndarray): True labels, shape (n_samples,).
        y_pred (np.ndarray): Predicted labels, shape (n_samples,).
        output_dir (str): Directory to save the class metrics.
    """
    classes = np.unique(y_true)
    metrics = []

    metrics_dict = {}

    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append(f'Class {cls}: Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1_score:.2f}')

        metrics_dict[cls] = [precision, recall, f1_score]

    output_file = os.path.join(output_dir, 'class_metrics.txt')
    with open(output_file, 'w') as f:
        for metric in metrics:
            f.write(metric + '\n')

    cls_names = list(metrics_dict.keys())
    precisions = [metrics_dict[cls][0] for cls in classes]
    recalls = [metrics_dict[cls][1] for cls in classes]
    f1_scores = [metrics_dict[cls][2] for cls in classes]


    # Plot Precision
    plt.figure(figsize=(10, 8))
    plt.barh(classes, precisions, color='b', align='center')
    plt.xlabel('Precision')
    plt.ylabel('Class')
    plt.title('Class-wise Precision')
    plt.yticks(classes)
    plt.savefig(os.path.join(output_dir, 'precision.png'))
    plt.close()

    # Plot Recall
    plt.figure(figsize=(10, 8))
    plt.barh(classes, recalls, color='g', align='center')
    plt.xlabel('Recall')
    plt.ylabel('Class')
    plt.title('Class-wise Recall')
    plt.yticks(classes)
    plt.savefig(os.path.join(output_dir, 'recall.png'))
    plt.close()

    # Plot F1 Score
    plt.figure(figsize=(10, 8))
    plt.barh(classes, f1_scores, color='r', align='center')
    plt.xlabel('F1 Score')
    plt.ylabel('Class')
    plt.title('Class-wise F1 Score')
    plt.yticks(classes)
    plt.savefig(os.path.join(output_dir, 'f1_score.png'))
    plt.close()

def evaluation(model: nn.Module, valid_dl: DataLoader, output_directory: str):
    """
    Evaluate the model on image classification and save various metrics and plots.
    
    Args:
        model (nn.Module): Model to evaluate.
        valid_dl (DataLoader): DataLoader for the validation or test dataset.
        output_directory (str): Path to output heatmaps, confusion matrices, and other metrics.
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

    # Class specific validation: 
    calculate_metrics(all_labels, all_preds, output_dir=output_directory)

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
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 'ResNet18', 'ResNet34','HCVIT', 'MobileNet'], help='Model name')
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--config_file', type=str, required=True, default='config.json', help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output confusion matrix and attention maps.")

    args = parser.parse_args()

    valid_dl = load_dataset(root_dir=args.root_dir, mode="val", batch_size=1) 
    
    print("[INFO] Evaluation dataset defined.")
    
    model_config = load_config(args.config_file)
    
    if args.model == "ViT":
        model = ViT.get_ViT(input_dim=(3, configs.img_height, configs.img_width),
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
        model = ResNet.get_ResNet18(num_classes=configs.num_class)
        print("[INFO] ResNet18 Model loaded with the following attributes:")
        print(f"[INFO] * Channels: {model.channels}")
        print(f"[INFO] * Layers: {model.num_layers}")
    elif args.model == "ResNet34":
        model = ResNet.get_ResNet34(num_classes=configs.num_class)
        print("[INFO] ResNet34 Model loaded with the following attributes:")
        print(f"[INFO] * Channels: {model.channels}")
        print(f"[INFO] * Layers: {model.num_layers}")
    elif args.model == "HCVIT": 
        model = HybridCNNTransformer.get_HCViT(cnn_output_size=model_config.get("cnn_output_size"),
                                               d_model=model_config.get("d_model"),
                                               patch_size=model_config.get("patch_size"),
                                               head=model_config.get("head"), 
                                               num_layers=model_config.get("num_layers"), 
                                               num_classes=configs.num_class)
        print("[INFO] HCViT loaded with the following attributes: ")
        print(f"[INFO] * CNN Output Shape: {model_config.get('cnn_output_size')}")
        print(f"[INFO] * d_model: {model_config.get('d_model')}")
        print(f"[INFO] * Patch size: {model_config.get('patch_size')}")
        print(f"[INFO] * Number of attention heads: {model_config.get('head')}")
        print(f"[INFO] * Number of layers: {model_config.get('num_layers')}")
    elif args.model == "MobileNet":
        model = MobileNet.get_MobileNet(num_of_classes=configs.num_class)
        print("[INFO] MobileNet loaded with defined parameters.")
    
    if args.model_save_path:
        print("[INFO] Model weights provided. Loading model weights.")
        model.load_state_dict(torch.load(args.model_save_path))

    evaluation(model, valid_dl=valid_dl, output_directory=args.output_dir)