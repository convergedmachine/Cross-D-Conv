import torch
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

def load_weights(path):
    """Load model weights from the specified path."""
    return torch.load(path, map_location=torch.device('cpu'))

def normalize(tensor):
    """Normalize a tensor image to be between 0 and 1."""
    tensor = tensor.clone()
    min_val = float(tensor.min())
    max_val = float(tensor.max())
    tensor = (tensor - min_val) / (max_val - min_val + 1e-5)
    return tensor

def visualize_filters(weights_3d, images_per_row=8, slice_index=3, save_path="enhanced_filter_visualization.png"):
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
    slice_index = slice_index if slice_index is not None else total_slices // 2

    # Calculate grid size
    grid_size = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    axes = axes.flatten()

    for idx in range(num_filters):
        ax = axes[idx]
        filter_slice = weights_3d[idx, :, slice_index, :, :]
        filter_slice = normalize(filter_slice)
        filter_image = filter_slice.permute(1, 2, 0).numpy()
        ax.imshow(filter_image)
        ax.axis('off')

    # Remove any unused subplots
    for idx in range(num_filters, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=72)
    plt.show()

# Load the weights
weights = load_weights("../_weights/CDConvR18_IN1K.pth")
weights_3d = weights['model']['conv1.weights_3d']  # Shape: [64, 3, 7, 7, 7]
print(f"Weights shape: {weights_3d.size()}")

# Visualize the filters
visualize_filters(weights_3d)

# Parameters
batch_size = weights_3d.size(0)  # 64
channels = weights_3d.size(1)    # 3
slice_size = weights_3d.size(2)  # 7

# Create a subplot figure with 8 rows and 8 columns
fig = sp.make_subplots(
    rows=8, cols=8,
    horizontal_spacing=0.01,
    vertical_spacing=0.01,
    specs=[[{'type': 'volume'} for _ in range(8)] for _ in range(8)]
)

# Plotting each volume
for batch_index in range(batch_size):
    row = (batch_index // 8) + 1
    col = (batch_index % 8) + 1
    for channel in range(channels):
        vol = weights_3d[batch_index, channel].numpy()
        X, Y, Z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
        fig.add_trace(
            go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=vol.flatten(),
                isomin=vol.min(),
                isomax=vol.max(),
                opacity=0.1,
                surface_count=5,
                colorscale='inferno',
                showscale=False
            ),
            row=row,
            col=col
        )

# Update layout to hide axis ticks and set camera view
for i in range(1, 65):
    fig.update_scenes(
        dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            aspectmode='data'
        ),
        row=(i - 1) // 8 + 1,
        col=(i - 1) % 8 + 1
    )

# Adjust overall layout
fig.update_layout(
    height=1600,  # 200 px per subplot row
    width=1600,   # 200 px per subplot column
    showlegend=False,
    title_text="3D Convolutional Layer Weights Visualization"
)

# Save the figure as an image file
pio.write_image(fig, "compact_volume_plot.png", scale=2)
