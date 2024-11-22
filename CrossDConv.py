import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(CrossDConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # 3D Convolutional Weights with Parameter Sharing
        self.weights_fft = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size, dtype=torch.cfloat))
        nn.init.kaiming_normal_(self.weights_fft.real, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weights_fft.imag, mode='fan_out', nonlinearity='relu')        

        # Shared Rotation Parameters
        self.rotation_params = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),  # Predicting 3 rotation axes and 3 rotation angles
            nn.BatchNorm2d(3)
        )

    def get_rotation_params(self, x):
        """
        Obtain shared rotation axes and angles based on input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                axis: Normalized rotation axes (batch_size, 3).
                angle: Rotation angles constrained between -π/4 and π/4 (batch_size, 1).
        """
        angles = self.rotation_params(x)  # Shape: (batch_size, 6, H, W)

        # Aggregate rotation parameters by averaging over spatial dimensions
        angles = angles.mean(dim=(2, 3))  # Shape: (batch_size, 3)

        # Combine angles using a weighted sum or another strategy
        angle = angles.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
        angle = torch.tanh(angle) * (torch.pi / 4)  # Constrain angles

        return angle

    def rotate_weights_fft(self, cos_theta):
        """
        Apply rotation matrices to the 3D convolutional weights using FFT-based phase rotation.

        Args:
            R (torch.Tensor): Rotation matrices of shape (batch_size, 3, 3).

        Returns:
            torch.Tensor: Rotated weights of shape (batch_size, out_channels, in_channels/groups, K, K, K).
        """
        batch_size = cos_theta.size(0)
        out_channels, in_channels_per_group, K, _, _ = self.weights_fft.size()

        # Create frequency coordinates once
        freq = torch.fft.fftfreq(K).to(self.weights_fft.device)
        torch.cuda.empty_cache()
        fx, fy, fz = torch.meshgrid(freq, freq, freq, indexing='ij')  # Each shape: (K, K, K)

        # Expand frequency grids to include batch dimension
        fx = fx.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: (batch_size, K, K, K)
        fy = fy.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: (batch_size, K, K, K)
        fz = fz.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: (batch_size, K, K, K)

        # Extract individual elements for phase shift calculation
        r00 = cos_theta.view(batch_size, 1, 1, 1)  # Shape: (batch_size, 1, 1, 1)
        r11 = cos_theta.view(batch_size, 1, 1, 1)  # Shape: (batch_size, 1, 1, 1)
        r22 = cos_theta.view(batch_size, 1, 1, 1)  # Shape: (batch_size, 1, 1, 1)

        # Compute phase shift with proper broadcasting
        phase_shift = torch.exp(-2j * torch.pi * (
            fx * r00 +
            fy * r11 +
            fz * r22
        )).unsqueeze(1).unsqueeze(2)   # Adjust unsqueeze to match weights_fft dimensions

        # Broadcast weights_fft for batch processing
        weights_fft_batched = self.weights_fft.unsqueeze(0).expand(batch_size, out_channels, in_channels_per_group, K, K, K)

        # Apply phase shift
        weights_fft_rotated = weights_fft_batched * phase_shift

        # Inverse FFT to get rotated weights in spatial domain
        rotated_weights = torch.fft.ifftn(weights_fft_rotated, dim=(-3,-2,-1)).real
        torch.cuda.empty_cache()

        return rotated_weights

    def forward(self, x):
        """
        Forward pass of the EnhancedCrossDConv module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor after convolution and multiscale feature extraction.
        """
        batch_size = x.size(0)

        # Obtain rotation parameters
        angles = self.get_rotation_params(x)
        cos_theta = torch.cos(angles)

        # Rotate convolutional weights using FFT-based rotation
        rotated_weights = self.rotate_weights_fft(cos_theta)  # Shape: (batch_size, out_channels, in_channels/groups, K, K, K)

        # Extract 2D kernels from the rotated 3D kernels (middle slice)
        middle_slice = self.kernel_size // 2
        twod_rotated_weights = rotated_weights[:, :, :, middle_slice, :, :]  # Shape: (batch_size, out_channels, in_channels/groups, K, K)

        # Reshape for group convolution
        grouped_weights = twod_rotated_weights.view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size
        )  # Shape: (batch_size*out, in_channels/groups, K, K)

        # Prepare input for group convolution
        x_grouped = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))  # Shape: (1, batch_size*in_channels, H, W)

        # Perform group convolution
        conv_output = F.conv2d(
            x_grouped,
            grouped_weights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=batch_size
        )  # Shape: (1, batch_size*out_channels, H_out, W_out)

        # Reshape output to (batch_size, out_channels, H_out, W_out)
        conv_output = conv_output.view(batch_size, self.out_channels, conv_output.size(2), conv_output.size(3))

        return conv_output

import time

def benchmark():
    # Define input dimensions
    batch_size = 45
    in_channels = 16
    out_channels = 32
    height, width = 224, 224
    depth = 32  # For Conv3d
    kernel_size = 7

    # Initialize inputs
    input_2d = torch.randn(batch_size, in_channels, height, width).cuda()
    input_3d = torch.randn(batch_size, in_channels, depth, 96, 96).cuda()

    # Initialize layers
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1).cuda()
    conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1).cuda()
    optimized_conv = CrossDConv(in_channels, out_channels, kernel_size).cuda()

    # Warm-up
    for _ in range(10):
        conv2d(input_2d)
        conv3d(input_3d)
        optimized_conv(input_2d)

    # Benchmark Conv2d
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_conv2d = conv2d(input_2d)
    torch.cuda.synchronize()
    end = time.time()
    conv2d_time = end - start

    # Benchmark Conv3d
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_conv3d = conv3d(input_3d)
    torch.cuda.synchronize()
    end = time.time()
    conv3d_time = end - start

    # Benchmark OptimizedCrossDConv
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_optimized = optimized_conv(input_2d)
    torch.cuda.synchronize()
    end = time.time()
    optimized_time = end - start

    print(f"Conv2d Time: {conv2d_time:.4f} seconds")
    print(f"Conv3d Time: {conv3d_time:.4f} seconds")
    print(f"OptimizedCrossDConv Time: {optimized_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
