# Sparsed-view Dynamic 3D Gaussians in the Wild
### [Project Page](https://dynamic3dgaussians.github.io/) | [Paper](https://arxiv.org/pdf/2308.09713.pdf) | [ArXiv](https://arxiv.org/abs/2308.09713) | [Tweet Thread](https://twitter.com/JonathonLuiten/status/1692346451668636100) | [Data](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip) | [Pretrained Models](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip)
Reimplementation of [InstantSplat](https://instantsplat.github.io/) <br><br>
[Sparsed-view Dynamic 3D Gaussians in the Wild](https://dynamic3dgaussians.github.io/)  
 [Juan Atehortúa](atehortua.me) <sup>1</sup>,
 [Alice Yu]() <sup>1</sup>
 <sup>1</sup> Massachusetts Institute of Technology
 TBD, 2024 <br>
ate@mit.edu

<p float="middle">
  <img src="./teaser_figure.png" width="99%" />
</p>

## Installation
```bash
# Install this repo (pytorch)
git clone https://github.com/Atehortuajf/Dynamic3DGaussians.git
conda env create --file environment.yml
conda activate dynamic_gaussians

# Install Gaussian Rasterizer
git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .

# Install Dust3r and CroCo
git clone --recursive https://github.com/naver/dust3r.git
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
