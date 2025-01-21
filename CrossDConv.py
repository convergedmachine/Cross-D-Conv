import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm

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


from acsconv.operators import ACSConv
HAVE_ACSCONV = True

def benchmark_single_pass(
    layer, 
    input_tensor, 
    warmup=10, 
    iters=100, 
    label="layer"
):
    """
    Times the forward pass of `layer` on `input_tensor`.
    Performs a warmup, then measures `iters` runs.
    Returns average runtime in seconds.
    Uses tqdm for a progress bar around the loop.
    """
    # Warm-up with tqdm
    for _ in tqdm(range(warmup), desc=f"{label} [Warmup]"):
        _ = layer(input_tensor)

    # Sync GPU before timing
    torch.cuda.synchronize()
    start = time.time()
    
    # Main benchmark loop with tqdm
    for _ in tqdm(range(iters), desc=f"{label} [Benchmark]"):
        _ = layer(input_tensor)
    
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iters
    print(f"{label}: {avg_time*1000:.3f} ms per iteration (avg over {iters} runs)")
    return avg_time


def benchmark_comparison():
    """
    Compare Conv2d, CrossDConv, ACSConv, and Conv3d 
    across various 2D and 3D input shapes.
    """
    import torch
    import torch.nn as nn
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------------
    # 1) Define multiple shapes for 2D experiments
    #    Format: (batch_size, in_channels, H, W)
    # ----------------------------------------------------------------
    shapes_2d = [
        (8, 64, 128, 128),
        (16, 128, 224, 224),
        (32, 384, 224, 224),
    ]

    # ----------------------------------------------------------------
    # 2) Define multiple shapes for 3D experiments
    #    Format: (batch_size, in_channels, D, H, W)
    # ----------------------------------------------------------------
    shapes_3d = [
        (4, 64, 16, 64, 64),
        (4, 64, 32, 96, 96),
        (8, 128, 16, 128, 128),
    ]

    # Hyperparameters for the benchmark
    kernel_size_2d = 3
    kernel_size_3d = 3
    out_channels_2d = 512
    out_channels_3d = 512
    
    # ----------------------------------------------------------------
    #  BENCHMARK: 2D Input
    # ----------------------------------------------------------------
    print("=========== 2D Input Benchmark ===========")
    for shape in shapes_2d:
        B, C, H, W = shape
        print(f"\n--- Shape: {shape} ---")
        x_2d = torch.randn(shape, device=device)
        
        # 2D conv
        conv2d = nn.Conv2d(
            in_channels=C,
            out_channels=out_channels_2d,
            kernel_size=kernel_size_2d,
            padding=1
        ).to(device)
        
        # CrossDConv (designed for 2D input, 3D-like kernel partitioning logic)
        crossd = CrossDConv(
            in_channels=C,
            out_channels=out_channels_2d,
            kernel_size=kernel_size_2d,
            padding=1,
            groups=1
        ).to(device)

        # 3D conv on 2D input => forcibly reshape input to (B, C, 1, H, W).
        conv3d_2dshaped = nn.Conv3d(
            in_channels=C,
            out_channels=out_channels_2d,
            kernel_size=kernel_size_3d,
            padding=1
        ).to(device)
        x_2d_as_3d = x_2d.unsqueeze(2)  # => (B, C, 1, H, W)
        
        # ACSConv (3D) forcibly on "fake 3D" input with depth=1
        if HAVE_ACSCONV:
            acs_2dshaped = ACSConv(
                in_channels=C,
                out_channels=out_channels_2d,
                kernel_size=kernel_size_3d,
                padding=1
            ).to(device)
        else:
            acs_2dshaped = None

        # Benchmark each
        _ = benchmark_single_pass(conv2d, x_2d, label="Conv2d")
        _ = benchmark_single_pass(crossd, x_2d, label="CrossDConv")
        _ = benchmark_single_pass(conv3d_2dshaped, x_2d_as_3d, label="Conv3d [2D->3D shape]")
        
        if acs_2dshaped is not None:
            _ = benchmark_single_pass(acs_2dshaped, x_2d_as_3d, label="ACSConv [2D->3D shape]")


if __name__ == "__main__":
    benchmark_comparison()
