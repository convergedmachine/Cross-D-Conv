import numpy as np
import pandas as pd

import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from acsconv.converters import ACSConverter
from torchvision.models import resnet18, ResNet18_Weights
from _models.replace_conv_layers import convert2threed
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    NormalizeIntensity
)
from sklearn.model_selection import KFold

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def freeze_all_but_bn(m):
    if not isinstance(m, nn.BatchNorm3d) and not isinstance(m, nn.Linear):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
    else:
        print("Weights not freezed!", m)

def create_model(conv, n_classes):
    if conv == 'acsconv':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
        model = ACSConverter(model)

    elif conv == 'acsconvrin':
        model = resnet18(num_classes=n_classes)
        weights = torch.load("../../checkpoints_regular/checkpoint.pth")["model"]
        del weights['fc.weight']
        del weights['fc.bias']        
        model.load_state_dict(weights, strict=False)
        model = ACSConverter(model)

    elif conv == 'crossdconv':    
        model = resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
        
        convert2threed(model)
        
        weights = torch.load("../../CrossDConvR18.pth", map_location='cpu')['model']
        del weights['fc.weight']
        del weights['fc.bias']

        filtered_params = {}

        for name, param in weights.items():
            # Check if the parameter name contains 'bn' (batch normalization) or 'rotation'
            if 'rotation' not in name and 'angle' not in name and 'dilated_conv' not in name and 'bn_dilated' not in name:
                filtered_params[name.replace('.weights_3d', '.weight')] = param

        log = model.load_state_dict(filtered_params, strict=False)    
        print(log)
        
    elif conv == 'random':    
        model = resnet18(num_classes=n_classes)
        convert2threed(model) 
        model.apply(reset_weights)

    #model.apply(freeze_all_but_bn)
    return model

def train_model_cls(model, train_loader, optimizer, loss_function, device):
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs.repeat(1, 3, 1, 1, 1))
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_model_cls(model, val_loader, device):
    model.eval()
    num_correct = 0.0
    total = 0

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images.repeat(1, 3, 1, 1, 1))
            _, preds = torch.max(val_outputs, dim=1)
            num_correct += torch.sum(preds == val_labels).item()
            total += val_labels.size(0)

    accuracy = num_correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

def train_model_reg(model, train_loader, optimizer, loss_function, device):
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs.repeat(1, 3, 1, 1, 1)).squeeze()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_model_reg(model, val_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            outputs = model(val_images.repeat(1, 3, 1, 1, 1)).squeeze()
            all_labels.extend(val_labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    mse = np.mean((all_labels - all_outputs) ** 2)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

def main(conv):
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read only the 'filename', 'sex', and 'age' columns from the CSV
    df = pd.read_csv('data/IXI.csv', usecols=['Filename', 'SEX', 'AGE'])

    images = [
        "IXI314-IOP-0889-T1.nii.gz",
        "IXI249-Guys-1072-T1.nii.gz",
        "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz",
        "IXI020-Guys-0700-T1.nii.gz",
        "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz",
        "IXI577-HH-2661-T1.nii.gz",
        "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz",
        "IXI607-Guys-1097-T1.nii.gz",
        "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz",
        "IXI344-Guys-0905-T1.nii.gz",
        "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz",
        "IXI253-HH-1694-T1.nii.gz",
        "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz",
        "IXI585-Guys-1130-T1.nii.gz",
    ]    
    
    #df = orig_df[orig_df['Filename'].isin(images)]

    images = df['Filename'].apply(lambda x: f"./data/ixi/{x}").to_numpy()
    
    if args.type == 'regression':
        y = df['AGE'].to_numpy().astype(np.float32)
        num_classes = 1
    elif args.type == 'classification':
        y = df['SEX'].to_numpy() - 1
        num_classes = 2
    else:
        raise ValueError("Invalid type. Choose 'regression' or 'classification'.")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    if conv in ['acsconv', 'random']:
        MEAN = np.mean(np.array([0.485, 0.456, 0.406]))
        STD = np.mean(np.array([0.229, 0.224, 0.225]))
    elif conv in ['crossdconv', 'acsconvrin']:
        MEAN = np.array([0.3162])
        STD = np.array([0.3213])

    # Define transforms
    train_transforms = Compose([
        ScaleIntensity(),
        NormalizeIntensity(subtrahend=MEAN, divisor=STD),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
        RandRotate90()
    ])
    val_transforms = Compose([
        ScaleIntensity(),
        NormalizeIntensity(subtrahend=MEAN, divisor=STD),
        EnsureChannelFirst(),
        Resize((96, 96, 96))
    ])

    # Apply 5-fold cross-validation and test set. 1 fold is used for testing, 3 folds for training and 1 fold for validation.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_val_index, test_index) in enumerate(kf.split(images)):
        print(f"\n=== Starting Fold {fold + 1} ===")
        train_val_images, test_images = images[train_val_index], images[test_index]
        train_val_y, test_y = y[train_val_index], y[test_index]
        
        # Further split train_val into train and validation sets (80% train, 20% validation)
        val_split = int(0.2 * len(train_val_images))
        val_images, train_images = train_val_images[:val_split], train_val_images[val_split:]
        val_y, train_y = train_val_y[:val_split], train_val_y[val_split:]
        
        # Create datasets and dataloaders
        train_ds = ImageDataset(image_files=train_images, labels=train_y, transform=train_transforms)
        val_ds = ImageDataset(image_files=val_images, labels=val_y, transform=val_transforms)
        test_ds = ImageDataset(image_files=test_images, labels=test_y, transform=val_transforms)
        
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=pin_memory)

        model = create_model(conv, num_classes).to(device)
        
        if args.type == 'classification':
            loss_function = nn.CrossEntropyLoss()
        elif args.type == 'regression':
            loss_function = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        if args.type == 'classification':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif args.type == 'regression':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        max_epochs = 25
        best_metric = -1 if args.type == 'classification' else float('inf')
        best_epoch = -1

        for epoch in range(max_epochs):
            print(f"\nFold {fold + 1}, Epoch {epoch + 1}/{max_epochs}")
            
            if args.type == 'classification':
                train_loss = train_model_cls(model, train_loader, optimizer, loss_function, device)
                current_metric = validate_model_cls(model, val_loader, device)
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), f"best_{args.type}_{args.conv}_model_fold{fold + 1}.pth")
                    print("Saved new best classification model")
                
                scheduler.step()
            
            elif args.type == 'regression':
                train_loss = train_model_reg(model, train_loader, optimizer, loss_function, device)
                rmse = validate_model_reg(model, val_loader, device)
                
                if rmse < best_metric:
                    best_metric = rmse
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), f"best_{args.type}_{args.conv}_model_fold{fold + 1}.pth")
                    print("Saved new best regression model")
                
                scheduler.step(rmse)
            
            print(f"Fold {fold + 1}, Epoch {epoch + 1} completed. Best metric: {best_metric:.4f} at epoch {best_epoch}")

        # Load the best model for evaluation
        model_path = f"best_{args.type}_{args.conv}_model_fold{fold + 1}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded best model from epoch {best_epoch} for Fold {fold + 1}")

        # Evaluate on the test set
        if args.type == 'classification':
            test_metrics = validate_model_cls(model, test_loader, device)
            test_accuracy = test_metrics
            print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.4f}")
        elif args.type == 'regression':
            model.eval()
            all_labels = []
            all_outputs = []
            with torch.no_grad():
                for test_data in test_loader:
                    test_inputs, test_labels = test_data[0].to(device), test_data[1].to(device)
                    outputs = model(test_inputs).squeeze()
                    all_labels.extend(test_labels.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())
            
            all_labels = np.array(all_labels)
            all_outputs = np.array(all_outputs)
            mse = np.mean((all_labels - all_outputs) ** 2)
            rmse = np.sqrt(mse)
            print(f"Fold {fold + 1} Test RMSE: {rmse:.4f}")

    print("\nCross-validation completed.")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regression and Classification Analyses on IXI Dataset')
    
    parser.add_argument('--type',
                        default='regression',
                        choices=['regression', 'classification'],
                        help='Choose task type: regression or classification',
                        type=str)
    
    parser.add_argument('--conv',
                        default='crossdconv',
                        choices=['acsconv', 'acsconvrin', 'crossdconv', 'random'],
                        help='Choose converter: acsconv, acsconvrin, crossdconv, or random',
                        type=str)

    args = parser.parse_args()   
    main(args.conv)