<div align="center">

# Dual-Query Multiple Instance Learning for Dynamic Meta-Embedding based Tumor Classification

[![BMVC 2023](https://img.shields.io/badge/British_Machine_Vision_Conference_2023_(Oral)-BMVC_2023-4b44ce.svg)](https://papers.bmvc2023.org/0575.pdf)
<br>
[![python](https://img.shields.io/badge/-Python_3.7-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3717/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.12-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
</div>

<div align="center">

## [Paper](https://papers.bmvc2023.org/0575.pdf) / [Project Page](https://proceedings.bmvc2023.org/575/)
</div>

This repository covers the official implementation of our Dual-Query multiple instance learning approach for histopathological image analysis, a [BMVC 2023 (Oral) paper](https://papers.bmvc2023.org/0575.pdf). A novel method for histopathological slide assessment, extending on the [perceiver](http://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf) architecture and leveraging a dynamic meta-embedding strategy.
<br><br>
[Dual-Query Multiple Instance Learning for Dynamic Meta-Embedding based Tumor Classification](https://papers.bmvc2023.org/0575.pdf)<br>
[Simon Holdenried-Krafft](https://www.grk2543.uni-stuttgart.de/en/team/Holdenried-Krafft/)<sup>1</sup>, [Peter Somers](https://www.grk2543.uni-stuttgart.de/en/team/Somers-00001/)<sup>3</sup>, [Ivonne A. Montes-Majarro](https://www.grk2543.uni-stuttgart.de/team/Montes-Mojarro/)<sup>2</sup>, [Diana Silimon](https://www.grk2543.uni-stuttgart.de/en/team/Silimon/)<sup>2</sup>, [Cristina Tarín](https://www.grk2543.uni-stuttgart.de/en/team/Tarin-Sauer-00003/)<sup>3</sup>, [Falko Fend](https://www.grk2543.uni-stuttgart.de/en/team/Fend/)<sup>2</sup>, [Hendrik P. A. Lensch](https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/computergrafik/computer-graphics/staff/prof-dr-ing-hendrik-lensch/)<sup>1</sup><br>
<sup>1</sup>University of Tübingen, <sup>2</sup>University Hospital of Tübingen, <sup>3</sup>University of Stuttgart
<br><br>
![](images/overview.jpg)

## Setup

A conda environment is used for dependency management

```
conda create -n dqmil python=3.7
conda activate dqmil
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## Running

The configuration is based on [hydra](https://hydra.cc/). The settings can be found in the "configs" folder.

To train the Dual-Query Perceiver run:

```
python ./src/train.py 
```

## Datasets

The implementation relies on lmdbs. The corresponding code to create the datasets will be released in the upcoming weeks. 

## Citation

If you find this code useful, please consider citing:

```
@inproceedings{Holdenried-Krafft_2023_BMVC,
author    = {Simon Holdenried-Krafft and Peter Somers and Ivonne Montes-Mojarro and Diana Silimon and Cristina Tarín and Falko Fend and Hendrik P. A. Lensch},
title     = {Dual-Query Multiple Instance Learning for Dynamic Meta-Embedding based Tumor Classification},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0575.pdf}
}
```
