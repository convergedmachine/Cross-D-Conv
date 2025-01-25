import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def normalize(tensor):
    """Normalize a tensor image to be between 0 and 1."""
    tensor = tensor.clone()
    min_val = float(tensor.min())
    max_val = float(tensor.max())
    tensor = (tensor - min_val) / (max_val - min_val + 1e-5)
    return tensor

def visualize_tensor(weights_3d, filename):
    """
    Visualize 3D convolutional filters in 2D.

    Args:
        weights_3d (torch.Tensor): Tensor of shape [64, 3, 7, 7, 7].
        images_per_row (int): Number of filter images per row.
        slice_index (int): The index of the slice to visualize.
        save_path (str): Path to save the visualization image.
    """
    num_filters = weights_3d.size(0)
    num_channels = weights_3d.size(1)
    total_slices = weights_3d.size(2)

    # Select the middle slice if not specified
    slice_index = total_slices // 2

    # Calculate grid size
    grid_size = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    axes = axes.flatten()

    for idx in range(num_filters):
        ax = axes[idx]
        filter_slice = weights_3d[idx, :, slice_index, :, :]
        filter_slice = normalize(filter_slice)
        filter_image = filter_slice.permute(1, 2, 0).cpu().numpy()
        ax.imshow(filter_image)
        ax.axis('off')

    # Remove any unused subplots
    for idx in range(num_filters, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=64)
    plt.close()

class CrossDConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super(CrossDConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # 3D Convolutional Weights
        self.weights_3d = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weights_3d, mode='fan_out', nonlinearity='relu')

        # Rotation parameters network
        self.rotation_params = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1),
            # Optionally add BatchNorm2d(4) again,
            # but be aware that your final 4 channels might represent 
            # k_x, k_y, k_z, angle, so you may prefer them unnormalized.
        )

    def get_rotation_params(self, x):
        """
        Predict dynamic axis (k_x, k_y, k_z) and angle (theta) from x.
        """
        B = x.size(0)
        # shape: (batch_size, 4, H, W)
        rot_map = self.rotation_params(x)

        # Aggregate over spatial dims => (batch_size, 4)
        # rot_map: (B, 4, H, W)
        spatial_weights = F.softmax(rot_map.view(B, 4, -1), dim=-1)  
        rot_vec = (rot_map.view(B, 4, -1) * spatial_weights).sum(dim=-1)

        # Split into (k_x, k_y, k_z) and angle
        k = rot_vec[:, 0:3]           # (batch_size, 3)
        angle = rot_vec[:, 3:4]       # (batch_size, 1)

        # Normalize axis: k = k / (||k|| + eps)
        norm_k = k.norm(dim=1, keepdim=True) + 1e-8
        k = k / norm_k

        # Constrain angle to [- pi/4, pi/4]
        angle = torch.tanh(angle) * (torch.pi / 4)

        return k, angle

    def approximate_rotation_matrix(self, k, angle):
        """
        Construct the batch of 3x3 rotation matrices using the linear approximation:
          R ~ I + theta * K
        where K is the skew-symmetric matrix from axis k.
        
        k: (batch_size, 3)
        angle: (batch_size, 1)
        Return: (batch_size, 3, 3)
        """
        batch_size = k.size(0)
        device = k.device

        # Build skew-symmetric matrix for each batch
        # kx, ky, kz: (batch_size,)
        kx, ky, kz = k[:,0], k[:,1], k[:,2]

        # K = [[ 0,  -kz,  ky ],
        #      [ kz,  0,  -kx ],
        #      [-ky,  kx,  0  ]]
        # shape => (batch_size, 3, 3)
        K = torch.zeros(batch_size, 3, 3, device=device)
        K[:,0,1] = -kz
        K[:,0,2] =  ky
        K[:,1,0] =  kz
        K[:,1,2] = -kx
        K[:,2,0] = -ky
        K[:,2,1] =  kx

        # R = I + theta * K
        I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3)
        angle_ = angle.view(batch_size,1,1)  # broadcast
        R = I + angle_ * K
        return R

    def rotate_weights_fft(self, k, angle):
        """
        Rotate 3D kernels via FFT with dynamic axis.
        """
        batch_size = k.size(0)
        out_ch, in_ch_per_group, Ksize, _, _ = self.weights_3d.size()

        # 1) FFT of original weights
        weights_fft = torch.fft.fftn(self.weights_3d, dim=(-3, -2, -1))

        # 2) Frequency grids
        freq = torch.fft.fftfreq(Ksize, d=1.0).to(self.weights_3d.device)
        fx, fy, fz = torch.meshgrid(freq, freq, freq, indexing='ij')  # (K, K, K)

        # Expand to (batch_size,K,K,K)
        fx = fx.unsqueeze(0).expand(batch_size, -1, -1, -1)
        fy = fy.unsqueeze(0).expand(batch_size, -1, -1, -1)
        fz = fz.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 3) Rotation matrix R for each batch
        R = self.approximate_rotation_matrix(k, angle)  # (batch_size, 3, 3)

        # Extract R entries
        r00 = R[:,0,0].view(batch_size,1,1,1)
        r01 = R[:,0,1].view(batch_size,1,1,1)
        r02 = R[:,0,2].view(batch_size,1,1,1)
        r10 = R[:,1,0].view(batch_size,1,1,1)
        r11 = R[:,1,1].view(batch_size,1,1,1)
        r12 = R[:,1,2].view(batch_size,1,1,1)
        r20 = R[:,2,0].view(batch_size,1,1,1)
        r21 = R[:,2,1].view(batch_size,1,1,1)
        r22 = R[:,2,2].view(batch_size,1,1,1)

        # 4) Rotate frequency coords f' = R f
        f_prime_x = fx*r00 + fy*r01 + fz*r02
        f_prime_y = fx*r10 + fy*r11 + fz*r12
        f_prime_z = fx*r20 + fy*r21 + fz*r22

        # 5) Phase shift = exp(-2πi (f'_x + f'_y + f'_z ))
        phase_shift = torch.exp(
            -2j * torch.pi * (f_prime_x + f_prime_y + f_prime_z)
        ).unsqueeze(1).unsqueeze(2)
        # => shape (batch_size, 1, 1, K, K, K)

        # 6) Broadcast weights_fft to batch
        weights_fft_batched = weights_fft.unsqueeze(0).expand(
            batch_size, out_ch, in_ch_per_group, Ksize, Ksize, Ksize
        )

        # 7) Apply rotation in frequency, then iFFT
        weights_fft_rotated = weights_fft_batched * phase_shift
        rotated_weights = torch.fft.ifftn(weights_fft_rotated, dim=(-3, -2, -1)).real
        return rotated_weights

    def forward(self, x):
        batch_size = x.size(0)

        # 1) Predict dynamic axis + angle
        k, angle = self.get_rotation_params(x)

        # 2) Rotate weights
        #rotated_weights = self.rotate_weights_fft(k, angle)
        
        return torch.cat([k, angle], axis=1)


def load_weights(path):
    """
    Load model weights from the specified path.
    Assumes the saved dictionary includes a 'model' key.
    """
    return torch.load(path)  # returns the entire checkpoint dict


# 1) Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Create a CrossDConv layer
layer = CrossDConv(3, 64, kernel_size=7, stride=2, padding=3).to(device)
layer.eval()

# 3) Load external weights from checkpoint
weights_3d = load_weights("./_weights/CDConvR18_IN1K.pth")['model']

# 4) Populate your layer's parameters from the checkpoint
layer.load_state_dict({
    "rotation_params.0.weight": weights_3d['conv1.rotation_params.0.weight'],
    "rotation_params.0.bias": weights_3d['conv1.rotation_params.0.bias'],
    "rotation_params.1.weight": weights_3d['conv1.rotation_params.1.weight'],
    "rotation_params.1.bias": weights_3d['conv1.rotation_params.1.bias'],
    "rotation_params.1.running_mean": weights_3d['conv1.rotation_params.1.running_mean'],
    "rotation_params.1.running_var": weights_3d['conv1.rotation_params.1.running_var'],
    "rotation_params.3.weight": weights_3d['conv1.rotation_params.3.weight'],
    "rotation_params.3.bias": weights_3d['conv1.rotation_params.3.bias'],
    "weights_3d": weights_3d['conv1.weights_3d']
})

from load_data import load_data
from get_args_parser import get_args_parser

args = get_args_parser().parse_args()
_, data_loader_test, _, _, _ = load_data(args)

outputs_list = []
label_list = []

# Typically, for inference, you disable gradient calculation:
with torch.no_grad():
    for data, label in tqdm(data_loader_test):
        data = data.to(device)
        output = layer(data)        # This should yield [B, 4] if B is the batch size
        outputs_list.append(output.cpu())  # Move back to CPU (if needed) and store
        label_list.append(label)  # Move back to CPU (if needed) and store

# After the loop, concatenate all batch results into [N, 4]
final_outputs = torch.cat(outputs_list, dim=0)
label_lists = torch.cat(label_list, dim=0)
ks, angles = final_outputs[:, 0:3], final_outputs[:, 3:4]

# Convert PyTorch tensors to NumPy arrays
ksnp = ks.detach().cpu().numpy()
anglesnp = angles.detach().cpu().numpy()

# Print out basic statistics for ks
print("=== ks statistics ===")
print(f"Min:     {ksnp.min()}")
print(f"Max:     {ksnp.max()}")
print(f"Mean:    {ksnp.mean()}")
print(f"Median:  {np.median(ksnp)}")
print(f"Std:     {ksnp.std()}")
print()

# Print out basic statistics for angles
print("=== angles statistics ===")
print(f"Min:     {anglesnp.min()}")
print(f"Max:     {anglesnp.max()}")
print(f"Mean:    {anglesnp.mean()}")
print(f"Median:  {np.median(anglesnp)}")
print(f"Std:     {anglesnp.std()}")
print()

# Plot histograms
plt.figure(figsize=(24, 5))

# Histogram of ks (flatten to combine all three columns into one distribution)
plt.subplot(1, 4, 1)
plt.hist(ksnp[:,0].flatten(), bins=50, color='blue', alpha=0.7)
plt.title('k Distribution')
plt.xlabel('k-x values')
plt.ylabel('Frequency')

plt.subplot(1, 4, 2)
plt.hist(ksnp[:,1].flatten(), bins=50, color='blue', alpha=0.7)
plt.title('k Distribution')
plt.xlabel('k-y values')
plt.ylabel('Frequency')

plt.subplot(1, 4, 3)
plt.hist(ksnp[:,2].flatten(), bins=50, color='blue', alpha=0.7)
plt.title('k Distribution')
plt.xlabel('k-z values')
plt.ylabel('Frequency')

# Histogram of angles (flatten in case shape is (N,1))
plt.subplot(1, 4, 4)
plt.hist(anglesnp.flatten(), bins=50, color='green', alpha=0.7)
plt.title('Angles Distribution')
plt.xlabel('Angle values')
plt.ylabel('Frequency')

plt.tight_layout()

# Save the figure as "save_fig.png"
plt.savefig('save_fig.png', dpi=300, bbox_inches='tight')
plt.show()