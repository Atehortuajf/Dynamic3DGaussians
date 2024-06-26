# Sparsed-view Dynamic 3D Gaussians in the Wild
### [Project Page]() | [Paper]() | [ArXiv]()
Reimplementation of [InstantSplat](https://instantsplat.github.io/) <br><br>
[Sparsed-view Dynamic 3D Gaussians in the Wild](https://dynamic3dgaussians.github.io/)  
 [Juan Atehortúa](atehortua.me) <sup>1</sup>,
 [Alice Yu]() <sup>1</sup>
 <sup>1</sup> Massachusetts Institute of Technology
 TBD, 2024 <br>
ate@mit.edu

## Installation
```bash
# Install this repo (pytorch)
git clone --recursive https://github.com/Atehortuajf/Dynamic3DGaussians.git
conda env create --file environment.yml
conda activate dynamic_gaussians

# Install Gaussian Rasterizer
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .

# Optional but highly recommended, compile curope stuff for dust3r
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

## How to train models
We provide a hydra config tool ```config/train.yaml``` to specify the folder to the training data and other hyperparameters. In the folder, the program expects multiple .mp4 videos. Once the configuration is set, simply run ```python train.py``` to run the training loop.

## Run visualizer on pretrained models
A config file named ```config/visualize.yaml``` exists to specify what experiment and sequence to visualize.


## Code Structure
The code is essentially a reimplementation of InstantSplat on top of the Dynamic 3D Gaussian code, with the addition of hydra to make it easier to experiment.


## Camera Coordinate System
This code uses uses the OpenCV camera co-ordinate system (same as COLMAP). This is different to the blender / standard NeRF camera coordinate system. The conversion code between the two can be found [here](https://github.com/NVlabs/instant-ngp/blob/9f6ab886306ce1f9b359d3856e8a1907ce8b8c8b/scripts/colmap2nerf.py#L350)



## Citation
```
tbd
```
