"""
Use Imagenet10 for validation: kaggle datasets download -d liusha249/imagenet10
"""
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import load_dataset
import torch.optim as optim
import optuna
import argparse
from model_loader import load_model_class
import configs

def load_model(args):
    model_class = load_model_class(args.model)
    model = model_class(num_classes=configs.num_class)
    return model

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}, Accuracy: {100.*correct/total}%')
    
    return model

def objective(trial, model, train_dataloader, valid_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW'])
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    trained_model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=5, device=device)
    
    correct = 0
    total = 0
    trained_model.eval()
    
    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def optimize_hyperparameters(model, train_dataloader, valid_dataloader, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model, train_dataloader, valid_dataloader), n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f'  Value: {trial.value}')
    print("  Params: ")
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    return trial.params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['ResNet18', 
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
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    train_dataloader = load_dataset(root_dir=args.root_dir, mode = "train")
    valid_dataloader = load_dataset(root_dir=args.root_dir, mode = "test")
    best_params = optimize_hyperparameters(model, train_dataloader, valid_dataloader, n_trials=50)
    print("Best Hyperparameters:", best_params)
    
    
    
    
    

