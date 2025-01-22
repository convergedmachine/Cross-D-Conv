# Cross-D Conv: Bridging 2D and 3D Medical Imaging Analysis

This repository introduces **Cross-D Conv**, a novel convolutional operation designed to bridge the dimensional gap between 2D and 3D medical imaging datasets. By leveraging the Fourier domain for phase shifting, Cross-D Conv enables seamless weight transfer between 2D and 3D convolutional operations. This method addresses the challenge of multimodal data scarcity by utilizing abundant 2D data to enhance 3D model performance effectively.

## Key Features

- **Cross-Dimensional Weight Transfer**: Facilitates smooth transfer between 2D and 3D convolutional weights.
- **Efficient Pretraining**: Utilizes abundant 2D datasets (e.g., RadImagenet) to enhance 3D model performance.
- **Multimodal Data Support**: Designed for both 2D images and 3D volumetric medical datasets across CT, MRI, and ultrasound modalities.
- **Superior Performance**: Demonstrates statistically significant improvements over traditional convolution methods in diverse datasets.

---

## Performance Metrics

### Table 1: ResNet18 Performance on Imagenet and RadImagenet
| Dataset | Model          | Precision (Macro) | Recall (Macro) | F1 (Macro) | Balanced Accuracy | Average Accuracy |
|---------|----------------|-------------------|----------------|------------|-------------------|------------------|
| IN1K    | Regular        | 0.6807           | 0.6693         | 0.6657     | 0.6693           | 0.6693          |
|         | **Cross-D Conv** | **0.6895**       | **0.6881**     | **0.6838** | **0.6881**       | **0.6881** ↑    |
| RIN     | Regular        | 0.5830           | 0.4989         | 0.5252     | 0.4989           | 0.8305          |
|         | **Cross-D Conv** | **0.5891**       | **0.5228**     | **0.5471** | **0.5228**       | **0.8374** ↑    |

### Table 2: Performance on Image Datasets
| Dataset | Method        | OrganC Mean ± Std (CT) | OrganS Mean ± Std (CT) | Brain Tumor Mean ± Std (MRI) | Brain Dataset Mean ± Std (MRI) | Breast Mean ± Std (US) | Breast Cancer Mean ± Std (US) | Average |
|---------|---------------|------------------------|-------------------------|------------------------------|---------------------------------|------------------------|-------------------------------|---------|
| IN1K    | 2D Conv       | 0.862 ± 0.006         | 0.708 ± 0.035          | 0.884 ± 0.011               | 0.305 ± 0.023                 | 0.819 ± 0.019          | 0.745 ± 0.024                | 0.720   |
|         | **Cross-D Conv** | **0.871 ± 0.007**   | **0.763 ± 0.008**      | **0.892 ± 0.010**           | **0.308 ± 0.026**             | **0.836 ± 0.021**      | **0.759 ± 0.022**            | **0.738** ↑ |
| RIN     | 2D Conv       | 0.842 ± 0.006         | 0.742 ± 0.008          | 0.902 ± 0.010               | 0.268 ± 0.023                 | 0.832 ± 0.021          | 0.762 ± 0.016                | 0.725   |
|         | **Cross-D Conv** | **0.848 ± 0.008**   | **0.743 ± 0.008**      | **0.910 ± 0.013**           | **0.283 ± 0.023**             | **0.835 ± 0.037**      | **0.747 ± 0.024**            | **0.728** |

### Table 3: Performance on Volumetric Datasets
| Dataset | Method        | Mosmed Mean ± Std (CT) | Lung Aden. Mean ± Std (CT) | Fracture Mean ± Std (CT) | BraTS21 Mean ± Std (MRI) | IXI Mean ± Std (MRI) | BUSV Mean ± Std (US) | Average |
|---------|---------------|------------------------|----------------------------|--------------------------|--------------------------|-----------------------|-----------------------|---------|
| IN1K    | ACS-Conv      | **0.523 ± 0.057**     | **0.532 ± 0.034**         | 0.456 ± 0.027           | 0.539 ± 0.030           | 0.542 ± 0.044        | 0.559 ± 0.079        | 0.525   |
|         | **Cross-D Conv** | 0.505 ± 0.068      | 0.513 ± 0.071             | **0.469 ± 0.027**       | **0.549 ± 0.031**       | **0.583 ± 0.059**    | **0.590 ± 0.064**    | **0.535** ↑ |
| RIN     | ACS-Conv      | 0.547 ± 0.072         | **0.548 ± 0.034**         | 0.471 ± 0.034           | 0.545 ± 0.041           | 0.555 ± 0.046        | **0.604 ± 0.063**    | 0.545   |
|         | **Cross-D Conv** | **0.557 ± 0.102**   | 0.529 ± 0.058             | **0.491 ± 0.032**       | **0.558 ± 0.044**       | **0.559 ± 0.050**    | 0.602 ± 0.066        | **0.549** |

---

## Usage

To train the model using distributed training:

```bash
torchrun --nproc_per_node=8 --standalone train.py \
    --data-path /path/to/radimagenet \
    --workers 32 \
    --batch-size 32 \
    --sync-bn \
    --rot
```

## Performance

The model achieves state-of-the-art performance in cross-dimensional medical image analysis, demonstrating superior feature quality assessment compared to conventional methods.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yavuz2024cross,
  title={Cross-D Conv: Cross-Dimensional Transferable Knowledge Base via Fourier Shifting Operation},
  author={Yavuz, Mehmet Can and Yang, Yang},
  journal={arXiv preprint arXiv:2411.02441},
  year={2024}
}
```

![Dynamic Filters](dyns_filter_grid.gif)

## License

This project is licensed under the MIT License.
