[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-spectral-spatial-dependent-global-learning/hyperspectral-image-classification-on-casi)](https://paperswithcode.com/sota/hyperspectral-image-classification-on-casi?p=a-spectral-spatial-dependent-global-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-spectral-spatial-dependent-global-learning/hyperspectral-image-classification-on-pavia)](https://paperswithcode.com/sota/hyperspectral-image-classification-on-pavia?p=a-spectral-spatial-dependent-global-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-spectral-spatial-dependent-global-learning/hyperspectral-image-classification-on-indian)](https://paperswithcode.com/sota/hyperspectral-image-classification-on-indian?p=a-spectral-spatial-dependent-global-learning)



## Overview
```
Deep learning techniques have been widely applied to hyperspectral image (HSI) classiﬁcation and have achieved great
success. However, the deep neural network model has a large parameter space and requires a large number of labeled data.
Deep learning methods for HSI classification usually follow a patchwise learning framework. Recently, a fast patch-free 
global learning (FPGA) architecture was proposed for HSI classification according to global spatial context information. 
However, FPGA has difficulty extracting the most discriminative features when the sample data is imbalanced. In this paper, 
a spectral-spatial dependent global learning (SSDGL) framework based on global convolutional long short-term memory (GCL) 
and global joint attention mechanism (GJAM) is proposed for insufficient and imbalanced HSI classification. In SSDGL, the 
hierarchically balanced (H-B) sampling strategy and the weighted softmax loss are proposed to address the imbalanced sample 
problem. To effectively distinguish similar spectral characteristics of land cover types, the GCL module is introduced to 
extract the long short-term dependency of spectral features. To learn the most discriminative feature representations, the 
GJAM module is proposed to extract attention areas. The experimental results obtained with three public HSI datasets show 
that the SSDGL has powerful performance in insufficient and imbalanced sample problems and is superior to other state-of-the-art methods.
```
## Survey Paper

+ This is an official implementation of SSDGL in our paper ["A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification"]().

## References
```
[1] Q. Zhu, W. Deng, Z. Zheng, Y. Zhong, Q. Guan, W. Lin, L. Zhang, and D. Li, 
“A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification”, 
IEEE Trans. Cybern., DOI:10.1109/TCYB.2021.3070577.
```
## To Cite SSDGL in Publications

+ Please cite the following reference:
+ Q. Zhu, W. Deng, Z. Zheng, Y. Zhong, Q. Guan, W. Lin, L. Zhang, and D. Li, 
“A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification”, 
IEEE Trans. Cybern., DOI:10.1109/TCYB.2021.3070577.
+ You can contact the e-mail dengweihuan@cug.edu.cn if you have further questions about the usage of codes and datasets.
+ For any possible research collaboration, please contact Prof. Qiqi Zhu (zhuqq@cug.edu.cn).

## Requirements:
```
pytorch >= 1.1.0
tensorboardX
opencv, skimage, sklearn, pillow, SimpleCV

SimpleCV install:
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
or
download folder from https://github.com/Z-Zheng/SimpleCV.git and run python setup.py install
```

## Prepare datasets

### 1. PaviaU
```python
image_mat_path='./pavia/PaviaU.mat'
gt_mat_path='./pavia/PaviaU_gt.mat'
```

### 2. Salinas
```python
image_mat_path='./salinas/Salinas_corrected.mat',
gt_mat_path='./salinas/Salinas_gt.mat',
```

### 3. Indian Pines
```python
image_mat_path='./IndianPines/Indian_pines_corrected.mat',
gt_mat_path='./IndianPines/Indian_pines_gt.mat',
```
## run experiments

### 1. PaviaU
```bash
./module/SSDGL.py Need to adjust the number of categories
bash scripts/SSDGL_1_0_pavia.sh

```

### 2. Salinas
```bash
./module/SSDGL.py Need to adjust the number of categories
bash scripts/SSDGL_1_0_salinas.sh

```

### 3. Indian Pines
```bash
./module/SSDGL.py Need to adjust the number of categories
bash scripts/SSDGL_1_0_indianpine.sh

```
### 4. GRSS2013(HOS)
```bash
./module/SSDGL.py Need to adjust the number of categories
bash scripts/SSDGL_1_0_HOS.sh
```

## Citation
+ If you extend or use this work, please cite the paper where it was introduced:
```
@ARTICLE{9440852,
  author={Zhu, Qiqi and Deng, Weihuan and Zheng, Zhuo and Zhong, Yanfei and Guan, Qingfeng and Lin, Weihua and Zhang, Liangpei and Li, Deren},
  journal={IEEE Transactions on Cybernetics}, 
  title={A Spectral-Spatial-Dependent Global Learning Framework for Insufficient and Imbalanced Hyperspectral Image Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TCYB.2021.3070577}}
```

## Dataset acquire
```
GRSS2013_HOS datasets
Baidu Drive 
Link: https://pan.baidu.com/s/1kPF5f857cJHH617TluOLqQ    Code: a3qc
```
## Acknowledgments
```
Our code is inspired by FreeNet
https://github.com/Z-Zheng/FreeNet
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/nshaud)
```
