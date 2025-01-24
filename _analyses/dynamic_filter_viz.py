import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    """
    A custom convolutional layer that:
      1) Predicts a rotation axis (k_x, k_y, k_z) and angle (theta) from the input.
      2) Rotates a learned 3D kernel in the frequency domain.
      3) Extracts a 2D slice from the rotated 3D kernel as the final convolutional filter.
    """
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

        # rotation_params now predicts (k_x, k_y, k_z, angle)
        self.rotation_params = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1),
            # Optionally add a final BatchNorm2d(4), but be aware that
            # these channels might represent (k_x, k_y, k_z, angle),
            # so you might prefer them "unnormalized".
        )

    def get_rotation_params(self, x):
        """
        Predict dynamic axis (k_x, k_y, k_z) and angle (theta) from x.
        Returns:
            k (Tensor): shape (batch_size, 3)
            angle (Tensor): shape (batch_size, 1)
        """
        B = x.size(0)
        # shape of rot_map: (batch_size, 4, H, W)
        rot_map = self.rotation_params(x)

        # Aggregate over spatial dimensions => (batch_size, 4)
        # We use a spatial softmax for weighting the contribution of each spatial location.
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
        
        Args:
            k (Tensor): shape (batch_size, 3)
            angle (Tensor): shape (batch_size, 1)
        Returns:
            R (Tensor): shape (batch_size, 3, 3)
        """
        batch_size = k.size(0)
        device = k.device

        # Build skew-symmetric matrix for each batch
        kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
        K = torch.zeros(batch_size, 3, 3, device=device)
        K[:, 0, 1] = -kz
        K[:, 0, 2] =  ky
        K[:, 1, 0] =  kz
        K[:, 1, 2] = -kx
        K[:, 2, 0] = -ky
        K[:, 2, 1] =  kx

        # R = I + theta * K
        I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3)
        angle_ = angle.view(batch_size, 1, 1)  # broadcast
        R = I + angle_ * K
        return R

    def rotate_weights_fft(self, k, angle):
        """
        Rotate 3D kernels via FFT with a dynamic axis and angle.

        Args:
            k (Tensor): shape (batch_size, 3)
            angle (Tensor): shape (batch_size, 1)
        Returns:
            rotated_weights (Tensor): shape (batch_size, out_channels, in_channels//groups, K, K, K)
        """
        batch_size = k.size(0)
        out_ch, in_ch_per_group, Ksize, _, _ = self.weights_3d.size()

        # 1) FFT of original weights (out_ch, in_ch_per_group, K, K, K)
        weights_fft = torch.fft.fftn(self.weights_3d, dim=(-3, -2, -1))

        # 2) Frequency grids
        freq = torch.fft.fftfreq(Ksize, d=1.0).to(self.weights_3d.device)
        fx, fy, fz = torch.meshgrid(freq, freq, freq, indexing='ij')  # each shape: (K, K, K)

        # Expand each to (batch_size, K, K, K)
        fx = fx.unsqueeze(0).expand(batch_size, -1, -1, -1)
        fy = fy.unsqueeze(0).expand(batch_size, -1, -1, -1)
        fz = fz.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 3) Rotation matrix R for each batch => shape (batch_size, 3, 3)
        R = self.approximate_rotation_matrix(k, angle)

        # Extract R entries, shape them for broadcasting
        r00 = R[:, 0, 0].view(batch_size, 1, 1, 1)
        r01 = R[:, 0, 1].view(batch_size, 1, 1, 1)
        r02 = R[:, 0, 2].view(batch_size, 1, 1, 1)
        r10 = R[:, 1, 0].view(batch_size, 1, 1, 1)
        r11 = R[:, 1, 1].view(batch_size, 1, 1, 1)
        r12 = R[:, 1, 2].view(batch_size, 1, 1, 1)
        r20 = R[:, 2, 0].view(batch_size, 1, 1, 1)
        r21 = R[:, 2, 1].view(batch_size, 1, 1, 1)
        r22 = R[:, 2, 2].view(batch_size, 1, 1, 1)

        # 4) Rotate frequency coords: f' = Rf
        f_prime_x = fx * r00 + fy * r01 + fz * r02
        f_prime_y = fx * r10 + fy * r11 + fz * r12
        f_prime_z = fx * r20 + fy * r21 + fz * r22

        # 5) Phase shift = exp(-2πi * (f'_x + f'_y + f'_z))
        phase_shift = torch.exp(-2j * math.pi * (f_prime_x + f_prime_y + f_prime_z))
        # shape => (batch_size, K, K, K)
        # We'll expand it to (batch_size, 1, 1, K, K, K) for broadcasting
        phase_shift = phase_shift.unsqueeze(1).unsqueeze(2)

        # 6) Broadcast weights_fft to batch
        weights_fft_batched = weights_fft.unsqueeze(0).expand(
            batch_size, out_ch, in_ch_per_group, Ksize, Ksize, Ksize
        )

        # 7) Apply rotation in frequency, then iFFT
        weights_fft_rotated = weights_fft_batched * phase_shift
        rotated_weights = torch.fft.ifftn(weights_fft_rotated, dim=(-3, -2, -1)).real
        return rotated_weights

    def forward(self, x):
        """
        Forward pass for dynamic 2D convolution. Predict (k, angle), rotate the 3D kernel,
        and extract a 2D slice.

        Args:
            x (Tensor): shape (batch_size, in_channels, H, W)
        Returns:
            twod_kernels (Tensor): shape (batch_size, out_channels, in_channels//groups, K, K)
        """
        # 1) Predict dynamic axis + angle
        k, angle = self.get_rotation_params(x)

        # 2) Rotate weights
        rotated_weights = self.rotate_weights_fft(k, angle)
        # shape: (batch_size, out_channels, in_channels//groups, K, K, K)

        # 3) Extract 2D kernel slice from the center of the 3D kernel
        mid_slice = self.kernel_size // 2
        return rotated_weights


def load_weights(path):
    """
    Load model weights from the specified path.
    Assumes the saved dictionary includes a 'model' key.
    """
    return torch.load(path)  # returns the entire checkpoint dict


# ========================== EXAMPLE USAGE ============================

# 1) Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Create a CrossDConv layer
layer = CrossDConv(3, 64, kernel_size=7, stride=2, padding=3).to(device)
layer.eval()

# 3) Load external weights from checkpoint
weights_3d = load_weights("../checkpoints_CDConv_imagenet/checkpoint.pth")['model']

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

# 5) Prepare an output directory
os.makedirs("out", exist_ok=True)

# Instead of discrete axes_list, we do a single interpolation from X->Y->Z
x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device)
y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=device)
z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device)

# We'll do a total of 64 steps: the first half from X->Y, the second half from Y->Z
num_steps = 64
t_values = torch.linspace(0, 2, num_steps, device=device)  # from 0 to 2

# For the angle, we can still do something like from -π/4 to +π/4 in 64 steps
angle_values = torch.linspace(-math.pi/4, math.pi/4, num_steps, device=device)

os.makedirs("out", exist_ok=True)

with torch.no_grad():
    for idx, (t, angle_) in enumerate(zip(t_values, angle_values)):
        # 1) Interpolate the axis k in a piecewise manner:
        if t <= 1.0:
            # Interpolate X->Y
            alpha = t  # goes from 0 to 1
            k = (1 - alpha) * x_axis + alpha * y_axis
        else:
            # Interpolate Y->Z
            alpha = t - 1.0  # goes from 0 to 1
            k = (1 - alpha) * y_axis + alpha * z_axis
        
        # 2) Normalize k
        k = k / (k.norm(dim=1, keepdim=True) + 1e-8)

        # 3) Make angle a tensor of shape (1, 1)
        angle_ = angle_.view(1, 1)

        # 4) Rotate the 3D weights for this (k, angle)
        rotated = layer.rotate_weights_fft(k, angle_)
        # shape => (1, out_ch, in_ch//groups, K, K, K)

        # 5) Visualize the entire 3D kernel
        #    Use the same 'visualize_tensor' helper.
        out_path = f"out/interp_{idx:02d}.png"
        visualize_tensor(rotated.squeeze(0), out_path)

print("Done! Check the 'out/' folder for the generated kernel visualizations transitioning X->Y->Z.")


