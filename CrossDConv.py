import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
        rotated_weights = self.rotate_weights_fft(k, angle)
        # => (batch_size, out_ch, in_ch_per_group, K, K, K)

        # 3) Extract 2D kernel slice
        mid_slice = self.kernel_size // 2
        twod_kernels = rotated_weights[:, :, :, mid_slice, :, :]  # shape => (batch, out_ch, in_ch//grp, K, K)

        # 4) Reshape for group conv
        # Reshape weights: (batch_size * out_channels, in_channels // groups, K, K)
        grouped_weights = twod_kernels.view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size
        )

        # Reshape input: (1, batch_size * in_channels, H, W)
        x_grouped = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))

        # Determine the correct number of groups
        groups = batch_size * self.groups

        # Perform grouped convolution with groups=batch_size * self.groups
        conv_output = F.conv2d(
            x_grouped,
            grouped_weights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=groups  # Corrected: groups=batch_size * self.groups
        )
        # shape => (1, batch_size*out_channels, H_out, W_out)

        # Reshape back to (batch_size, out_channels, H_out, W_out)
        conv_output = conv_output.view(batch_size, self.out_channels, conv_output.size(2), conv_output.size(3))

        return conv_output

# -------------------------------------------------------------------
# Example Benchmark Code (updated)
# -------------------------------------------------------------------
def benchmark():
    # Define input dimensions
    batch_size = 45
    in_channels = 384  # Updated to match the error context
    out_channels = 512  # Example value; adjust as needed
    height, width = 224, 224
    depth = 32  # For Conv3d
    kernel_size = 3  # Adjusted for the error context

    # Initialize inputs
    input_2d = torch.randn(batch_size, in_channels, height, width).cuda()
    input_3d = torch.randn(batch_size, in_channels, depth, 96, 96).cuda()

    # Initialize layers
    # Calculate groups such that in_channels // groups is an integer
    # For in_channels=384 and desired in_channels_per_group=3, groups=128
    groups = 128

    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1).cuda()
    conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1).cuda()
    optimized_conv = CrossDConv(in_channels, out_channels, kernel_size, padding=1, groups=groups).cuda()

    print("Original 2D Input Shape:", input_2d.shape)
    print("Original 3D Input Shape:", input_3d.shape)
    print("Processed 2D Output Shape:", optimized_conv(input_2d).shape)
    
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
    print(f"CrossDConv (Small-Angle Approx.) Time: {optimized_time:.4f} seconds")


if __name__ == "__main__":
    benchmark()
