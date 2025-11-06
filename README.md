<h1 align="center"><strong>ReAct-GS</strong></h1>
<h2 align="center">Re-Activating Frozen Primitives for 3D Gaussian Splatting</h2>

<p align="center">
  <a href="">Yuxin Cheng</a> ·
  <a href="">Binxiao Huang</a> ·
  <a href="">Wenyong Zhou</a> ·
  <a href="">Taiqiang Wu</a> ·
  <a href="">Graziano Chesi</a> ·
  <a href="">Zhengwu Liu</a> ·
  <a href="">Ngai Wong</a><sup>*</sup>
  
</p>

<p align="center">Department of EEE, The University of Hong Kong, Pokfulum, Hong Kong SAR</p>

<p align="center"><sup>*</sup>corresponding authors</p>

<h3 align="center"><a href="https://dl.acm.org/doi/10.1145/3746027.3754958">Paper</a> | <a href="https://arxiv.org/pdf/2510.19653">arXiv</a> | <a href="https://react-gs.github.io/">Project Page</a></h3>

<p align="center">
  <img src="./assets/3dgs.png" alt="3DGS" width="49.5%">
  <img src="./assets/reactgs.png" alt="ReAct-GS" width="49.5%">
</p>
<p align="center">
<strong>3DGS (left)</strong> tends to exhibit blurring and needle-like artifacts, while <strong>ReAct-GS (right)</strong> effectively re-activates frozen primitives through proposed re-activation stategies to improve rendering quality. 
</p>

## Overview

**ReAct-GS** addresses a critical limitation in 3D Gaussian Splatting (3DGS): the problem of frozen primitives that become stuck in suboptimal configurations during training, causing over-reconstruction artifacts. Our method introduces targeted reactivation strategies: 1) **Importance-aware Densification (IAD)**; 2) **Density-guided Clone (DGC)**; 3)**Needle-shape Perturbance (NSP)**, a novel approach that periodically identifies and re-activates forzen Gaussian primitives, enabling them to contribute more effectively to scene reconstruction.

## Installation

### Environment Setup

We recommend using conda to create a virtual environment:

```bash
conda create -n reactgs python=3.9 -y
conda activate reactgs
```

### Install PyTorch

Install PyTorch with CUDA support:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Build Submodules

Build the differential Gaussian rasterization and simple-knn modules:

```bash
cd submodules
pip install -e ./diff-gaussian-rasterization
pip install -e ./simple-knn
```

## Dataset Preparation

The dataset preparation is aligned with [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 

### Mip-NeRF 360 Dataset

1. Download the Mip-NeRF 360 dataset from [the official website](https://jonbarron.info/mipnerf360/)
2. Combine the scenes into the following structure:

```
ReActGS-main
|---data_ms360
    |---bicycle
    |   |---images
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |   |---images_2
    |   |---images_4
    |   |---images_8
    |   |---sparse
    |       |---0
    |           |---cameras.bin
    |           |---images.bin
    |           |---points3D.bin
    |---bonsai
    |---counter
    |---garden
    |---kitchen
    |---room
    |---stump
    |---...
```

## Training

### Quick Start with Batch Scripts

To train on a single scene:

```bash
python train.py -s <path_to_scene> -m <output_path> -r 4(outdoor scenes)/2(indoor scenes)
```

#### Key Training Parameters

- `--nsp_interval`: Interval for applying Needle Shape Perturbance
- `--nsp_util`: Training iteration until which NSP is applied
- `--iterations`: Total number of training iterations
- `--densify_from_iter`: Start densification from this iteration
- `--densify_until_iter`: Stop densification at this iteration
- `--densification_interval`: Interval for densification operations

#### Example Commands

```bash
# Train with custom NSP parameters
python train.py -s data/flowers -m output/flowers -r 4
python train.py -s data/bonsai -m output/bonsai -r 2
```

## Rendering

### Render Test Views

```bash
python render.py -m <path_to_trained_model>
```

#### Options

- `--iteration`: Specify which checkpoint to use (default: -1, uses the final model)
- `--skip_train`: Skip rendering training views
- `--skip_test`: Skip rendering test views

## Evaluation

### Evaluate a Single Scene

```bash
python metrics.py -m <path_to_trained_model>
```

## Interactive Viewer

ReAct-GS is fully compatible with the original 3DGS viewer since our rendering process and point cloud storage format are identical to 3DGS.

Please refer to the [3DGS Interactive Viewers documentation](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) for detailed instructions on using the real-time viewer.


## Citation

If you find ReAct-GS useful for your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3746027.3754958,
  author = {Yuxin Cheng, Binxiao Huang, Taiqiang Wu, Wenyong Zhou, Zhengwu Liu, Graziano Chesi, Ngai Wong},
  title = {Re-Activate Frozen Primitive for 3D Gaussian Splatting},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3754958},
  doi = {10.1145/3746027.3754958},
  booktitle = {Proceedings of the 33nd ACM International Conference on Multimedia},
  location = {Dublin, Ireland},
  series = {MM '25'}
}
```

## Acknowledgements

This project is inspired and built upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Pixel-GS](https://pixelgs.github.io) and [Abs-GS](https://ty424.github.io/AbsGS.github.io/). We thank the authors for their excellent work and for making their code publicly available. Please follow the license terms of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) when using this code.

## License

This project is licensed under the same terms as the original 3D Gaussian Splatting. See [LICENSE.md](LICENSE.md) for details. This software is free for non-commercial, research and evaluation use.