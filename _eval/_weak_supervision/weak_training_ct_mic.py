import argparse
import os
from copy import deepcopy
from tqdm import tqdm
from monai.data import ArrayDataset, DataLoader
from medmnist import INFO
import torch
import torch.nn as nn
import torch.utils.data as data
from acsconv.converters import ACSConverter
from torchvision.models import resnet18, ResNet18_Weights
from _models.replace_conv_layers import convert2threed
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    ScaleIntensity,
    RandFlip,
    EnsureType
)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

MEAN = np.array([0.3162])
STD = np.array([0.3213])

# Define classification-specific transforms
train_transform_rin = Compose([
    ScaleIntensity(),
    NormalizeIntensity(subtrahend=MEAN, divisor=STD),
    RandFlip(
        spatial_axis=0,
        prob=0.5
    ),
    EnsureType()
])

eval_transform_rin = Compose([
    ScaleIntensity(),
    NormalizeIntensity(subtrahend=MEAN, divisor=STD),  # Standard normalization
    EnsureType()
])

MEAN = np.mean(np.array([0.485, 0.456, 0.406]))
STD = np.mean(np.array([0.229, 0.224, 0.225]))

# Define classification-specific transforms
train_transform_in1k = Compose([
    ScaleIntensity(),
    NormalizeIntensity(subtrahend=MEAN, divisor=STD),
    RandFlip(
        spatial_axis=0,
        prob=0.5
    ),
    EnsureType()
])

eval_transform_in1k = Compose([
    ScaleIntensity(),
    NormalizeIntensity(subtrahend=MEAN, divisor=STD),  # Standard normalization
    EnsureType()
])

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
    if conv =='acsconv':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
        model = ACSConverter(model)

    elif conv =='acsconvrin':
        model = resnet18(num_classes=n_classes)
        weights = torch.load("../../checkpoints_regular/checkpoint.pth")["model"]
        del weights['fc.weight']
        del weights['fc.bias']        
        model.load_state_dict(weights, strict=False)
        model = ACSConverter(model)

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
        
    elif conv =='random':    
        model = resnet18(num_classes=n_classes)
        convert2threed(model) 
        model.apply(reset_weights)

    model.apply(freeze_all_but_bn)
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
    
    return model

def test(model, data_loader, device):
    total_loss = []
    y_score = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
        
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            y_score = torch.cat((y_score, outputs), dim=0)
            y_true = torch.cat((y_true, targets), dim=0)      
        acc = accuracy_score(y_true.cpu().numpy(), torch.argmax(y_score, dim=1).cpu().numpy())
    return total_loss, acc

def train_model(conv, num_classes, train_loader, val_loader, num_epochs, device):
    model = create_model(conv, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)
    
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    
    for _ in tqdm(range(num_epochs), desc='Epochs', total=num_epochs):
        # Train the model
        model = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        # Evaluate the model
        val_loss, val_acc = test(model, val_loader, device)
        val_loss = sum(val_loss) / len(val_loss)

        # Deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = deepcopy(model.state_dict())
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

def main(data_flag, num_epochs, batch_size, size, conv, download, as_rgb, split_ratio):
    info = INFO[data_flag]
    n_classes = len(info['label'])
    device = torch.device('cuda:0')

    if conv =='acsconv' or conv =='random':
        train_transform = train_transform_in1k
        eval_transform = eval_transform_in1k
    elif conv =='crossdconv' or conv =='acsconvrin':
        train_transform = train_transform_rin
        eval_transform = eval_transform_rin
    
    # Load the data
    data = np.load(f'/home/myavuz/.medmnist/{data_flag}.npz')

    volumes = np.concatenate((data['train_images'], data['val_images'], data['test_images']), axis=0)
    labels = np.concatenate((data['train_labels'], data['val_labels'], data['test_labels']), axis=0)

    # if volume is 1 or no channel, copy it to 3 channels
    if volumes.shape[1] == 1:
        volumes = np.repeat(volumes, 3, axis=1)
    elif len(volumes.shape) == 4:
        volumes = np.expand_dims(volumes, axis=1)
        volumes = np.repeat(volumes, 3, axis=1)
    
    print(f"Data shape: {volumes.shape}")

    best_val_metrics = []
    best_test_metrics = []    
    for _ in range(3):
        # Apply 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_indices = list(kf.split(volumes))
        
        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            print(f"Fold {fold + 1}")

            # Split the data into training and testing sets for this fold
            train_volumes, test_volumes = volumes[train_idx], volumes[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]

            # Further split the training data into training and validation sets
            val_split = int(len(train_volumes) * 0.2)
            val_volumes, train_volumes = train_volumes[:val_split], train_volumes[val_split:]
            val_labels, train_labels = train_labels[:val_split], train_labels[val_split:]

            print(f"Train shape: {train_volumes.shape}, Val shape: {val_volumes.shape}, Test shape: {test_volumes.shape}")

            # Create datasets
            train_dataset = ArrayDataset(img=train_volumes, labels=train_labels, img_transform=train_transform)
            val_dataset = ArrayDataset(img=val_volumes, labels=val_labels, img_transform=eval_transform)
            test_dataset = ArrayDataset(img=test_volumes, labels=test_labels, img_transform=eval_transform)

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            best_model = train_model(conv, n_classes, train_loader, val_loader, num_epochs, device)
            val_metrics = test(best_model, val_loader, device)
            test_metrics = test(best_model, test_loader, device)
            best_val_metrics.append(val_metrics[1])
            best_test_metrics.append(test_metrics[1])
            
    mean_test_acc = np.mean(best_test_metrics)
    mean_test_std = np.std(best_test_metrics)

    test_log1 = 'test  acc: %.5f\n' % (mean_test_acc)
    test_log2 = 'test  std: %.5f\n' % (mean_test_std)

    log = '%s\n' % (data_flag) + test_log1 + test_log2 + '\n'
    print(log)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='organmnist3d',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=28,
                        help='the image size of the dataset, 28 or 64, default=28',
                        type=int)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--conv',
                        default='crossdconv',
                        help='choose converter from acsconv, crossdconv',
                        type=str)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--split_ratio',
                        type=float,
                        default=1.0,
                        help='split ratio of train_subset')    

    args = parser.parse_args()
    data_flag = args.data_flag
    num_epochs = args.num_epochs
    size = args.size
    batch_size = args.batch_size
    conv = args.conv
    download = args.download
    as_rgb = args.as_rgb
    split_ratio = args.split_ratio
    
    main(data_flag, num_epochs, batch_size, size, conv, download, as_rgb, split_ratio)
