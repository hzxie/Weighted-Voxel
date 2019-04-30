# Weighted-Voxel

This repository contains the source codes for the paper [Weighted Voxel: a novel voxel representation for 3D reconstruction (Xie et al. 2018)](https://dl.acm.org/citation.cfm?id=3240888).

## Abstract

3D reconstruction has been attracting increasing attention in the past few years. With the surge of deep neural networks, the performance of 3D reconstruction has been improved significantly. However, the voxel reconstructed by extant approaches usually contains lots of noise and leads to heavy computation. In this paper, we define a new voxel representation, named Weighted Voxel. It provides more abundant information, facilitating the subsequent learning and generalization steps. Unlike regular voxel which consists of zero-one, the proposed Weighted Voxel makes full use of the structure information of voxels. Experimental results demonstrate that Weighted Voxel not only performs better in reconstruction but also takes less time in training.

## Cite this work

If you find this work useful in your research, please consider citing:

```
@inproceedings{xie2018weighted,
    title={Weighted Voxel: a novel voxel representation for 3D reconstruction},
    author={Xie, Haozhe and Yao, Hongxun and Sun, Xiaoshuai and Zhou, Shangchen and Tong, Xiaojun},
    booktitle={International Conference on Internet Multimedia Computing and Service {ICIMCS} 2018},
    year={2018},
    organization={ACM}
}
```

## Project Page

The project page is available at [https://haozhexie.com/project/weighted-voxel](https://haozhexie.com/project/weighted-voxel).

## Overview

#### Overview of Weighted Voxels

![Overview of Weighted Voxels](https://infinitescript.com/wordpress/wp-content/uploads/2018/02/Weighted-Voxels-Overview.jpg)

The generation of Weighted Voxels can be regarded as applying a convolutional kernel on regular voxels.
The value of each voxel in the Weighted Voxel is weighted summed over voxel values of its immediate neighbors.
More formally, the value in Weighted Voxel can be calculated as

![Weighted Voxel Eqn](https://latex.codecogs.com/svg.latex?y_%7B%28i%2C%20j%2C%20k%29%7D%20%3D%20-%20%5Comega%20%28-1%29%5E%7B%5Cupsilon_%7B%28i%2C%20j%2C%20k%29%7D%7D%20-%20%5Csum_%7Bm%20%3D%20i%20-%201%7D%5E%7Bi%20&plus;%201%7D%5Csum_%7Bn%20%3D%20j%20-%201%7D%5E%7Bj%20&plus;%201%7D%5Csum_%7Bp%20%3D%20k%20-%201%7D%5E%7Bk%20&plus;%201%7D%20%28-1%29%5E%7B%5Cupsilon_%7B%28m%2C%20n%2C%20p%29%7D%7D)

where ![upsilon_i_j_k_in_0_1](https://latex.codecogs.com/svg.latex?%5Cupsilon_%7B%28i%2C%20j%2C%20k%29%7D%20%5Cin%20%5C%7B0%2C%201%5C%7D) denotes the value in the regular voxel, and ![omega](https://latex.codecogs.com/svg.latex?%5Comega) is set to 26. Specially, we define ![upsilon_i_j_k_eq_0](https://latex.codecogs.com/svg.latex?%5Cupsilon_%7B%28i%2C%20j%2C%20k%29%7D%20%3D%200) when ![i_eq_ne_1](https://latex.codecogs.com/svg.latex?i%20%3D%20-1), ![j_eq_ne_1](https://latex.codecogs.com/svg.latex?j%20%3D%20-1) or ![k_eq_ne_1](https://latex.codecogs.com/svg.latex?k%20%3D%20-1).

#### Methods Illustration

![Methods Illustration](https://infinitescript.com/wordpress/wp-content/uploads/2018/02/Weighted-Voxel-Methods-Illustration.jpg)

The network architecture of 3D-R2N2 and Weighted Voxel. Both of them consist of an encoder, a 3D convolutional LSTM, and a decoder.
In 3D-R2N2, the reconstructed voxels are composed of zeros and ones, while in Weighted Voxel, the voxel values are filled with integers.

#### Comparison of Reconstruction Results with 3D-R2N2

<img src="https://infinitescript.com/wordpress/wp-content/uploads/2018/02/Weighted-Voxel-Reconstruction-Samples.jpg" alt="Reconstruction Samples" width="640">

Reconstruction samples of (a) cars (b) cabinets (c) speakers (d) sofas on the ShapeNet testing dataset.
The Weighted Voxel preserves more structural details of 3D objects

## Prerequisites

### Install `pygpu`

Please follow the instruction on the [homepage of gpuarray](http://deeplearning.net/software/libgpuarray/installation.html).

### Install Other Python Denpendencies

```
pip3 install -r requirements.txt
```

### Create `.theanorc`

Please paste following lines to `~/.theanorc`:

```
[cuda]
root = /opt/cuda    # Please change it with your CUDA installation path

[global]
device = cuda0      # Please change it with your GPU device ID
floatX = float32
```
### Download Dataset

Use following commands to download the ShapeNet dataset:

```
cd /path/to/the/repository
mkdir -p datasets/ShapeNet
cd datasets/ShapeNet

wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
tar -xf ShapeNetRendering.tgz

wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
tar -xf ShapeNetVox32.tgz
```

Use following command to generate Weighted Voxel dataset:

```
cd /path/to/the/repository
python utils/binvox_weighting.py datasets/ShapeNet/ShapeNetVox32 datasets/ShapeNet/WeightedShapeNetVox32
```

## Get Started

Use following command to train the neural network:

```
python3 runner.py 
```

Use following command to test the neural network:

```
python3 runner.py \
      --test \
      --weights output/weights.npy
```

The pretrained model can be downloaded from [here](https://gateway.infinitescript.com/?fileName=Weighted-Voxel.npy) (206 MB).

## License

This project is open sourced under MIT license.
