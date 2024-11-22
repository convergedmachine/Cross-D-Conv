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
    RandFlip,
    Resize,
    ScaleIntensity,
    NormalizeIntensity
)
from sklearn.model_selection import KFold
import os

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
        log = model.load_state_dict(weights, strict=False)
        model = ACSConverter(model)
        print(log)

    elif conv =='crossdconv':    
        model = resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
        convert2threed(model)
        
        weights = torch.load("../../CrossDConvR18.pth", weights_only=True)
        del weights['fc.weight']
        del weights['fc.bias']

        filtered_params = {}

        for name, param in weights.items():
            # Check if the parameter name contains 'bn' (batch normalization) or 'rotation'
            if 'bn' not in name and 'rotation' not in name and 'angle' not in name and 'dilated_conv' not in name and 'bn_dilated' not in name:
                filtered_params[name.replace('.weights_3d', '.weight')] = param

        log = model.load_state_dict(filtered_params, strict=False)    
        print(log)
        
        model.fc = nn.Linear(num_features, n_classes)
        
    elif conv == 'random':    
        model = resnet18(num_classes=n_classes)
        convert2threed(model) 
        model.apply(reset_weights)

    model.apply(freeze_all_but_bn)
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
    return accuracy

def create_dataset_dict(csv_path, data_dir, modality):
    # Read the labels CSV file
    labels_df = pd.read_csv(csv_path)
    ids = labels_df['BraTS21ID'].values
    
    # Initialize dictionary for images and labels
    image_dict = {}
    
    # Create image paths and get corresponding labels
    for id in ids:
        # Format ID with leading zeros to make it 5 digits
        formatted_id = str(id).zfill(5)
        
        # Create the image path
        image_path = os.path.join(data_dir, formatted_id, f"{formatted_id}_{modality}.nii.gz")
        
        # Verify if the file exists
        if os.path.exists(image_path):
            # Store the label as a single value, not as an array
            image_dict[image_path] = labels_df[labels_df['BraTS21ID'] == id]['MGMT_value'].values[0]
    
    return image_dict

def main(conv, modality):
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Example usage
    csv_path = './data/train_labels.csv'
    data_dir = './data/output'
    num_classes = 2

    # Get image paths and labels
    image_dict = create_dataset_dict(csv_path, data_dir, modality)

    # list all image_paths and labels
    images = np.array(list(image_dict.keys()))
    y = np.array(list(image_dict.values()))

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
        RandFlip(
            spatial_axis=0,
            prob=0.5
        ),
    ])
    val_transforms = Compose([
        ScaleIntensity(),
        NormalizeIntensity(subtrahend=MEAN, divisor=STD),
        EnsureChannelFirst(),
    ])

    test_metrics = []
    for _ in range(3):    
        # Apply 5-fold cross-validation and test set. 1 fold is used for testing, 3 folds for training and 1 fold for validation.
        kf = KFold(n_splits=5, shuffle=True)

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
            
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=pin_memory)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=pin_memory)
            
            loss_function = nn.CrossEntropyLoss()

            max_epochs = 50
            best_metric = -1
            best_epoch = -1

            model = create_model(conv, num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * max_epochs, 0.75 * max_epochs], gamma=0.1)
            
            for epoch in range(max_epochs):
                print(f"\nFold {fold + 1}, Epoch {epoch + 1}/{max_epochs}")
                
                train_loss = train_model_cls(model, train_loader, optimizer, loss_function, device)
                current_metric = validate_model_cls(model, val_loader, device)
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), f"out/best_{modality}_{conv}_model_fold{fold + 1}.pth")
                    print("Saved new best classification model")
                
                scheduler.step()
                
                print(f"Fold {fold + 1}, Epoch {epoch + 1} completed. Best metric: {best_metric:.4f} at epoch {best_epoch}")

            # Load the best model for evaluation
            model_path = f"out/best_{modality}_{conv}_model_fold{fold + 1}.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"\nLoaded best model from epoch {best_epoch} for Fold {fold + 1}")

            # Evaluate on the test set
            test_accuracy = validate_model_cls(model, test_loader, device)
            print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.4f}")
            test_metrics.append(test_accuracy)

    print("\n=== Cross-validation completed ===")
    print(f"Average Test Mean Metric: {np.mean(test_metrics):.4f}")
    print(f"Average Test Std Metric: {np.std(test_metrics):.4f}")    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classification Analyses on Brain Tumor Dataset')
    
    parser.add_argument('--modality',
                        default='FLAIR',
                        choices=['T1w', 'T1wCE', 'T2w', 'FLAIR'],
                        help='Choose modality: t1w, t1wce, t2w, or flair',
                        type=str)
    
    parser.add_argument('--conv',
                        default='crossdconv',
                        choices=['acsconv', 'acsconvrin', 'crossdconv', 'random'],
                        help='Choose converter: acsconv, acsconvrin, crossdconv, or random',
                        type=str)

    args = parser.parse_args()   
    main(args.conv, args.modality)