# Image-classification-Pytorch

## Overview
This repository is built for educational purposes on image classification using various computer vision models. The main goal is to demonstrate how these models can be applied to classify images and perform compared to contemporary models on widely known datasets. Much of the novel mechanisms developed and published by others (hopefully my own in the future as well) are reimplemented on this repository to demonstrate the intuition and advantages behind such mechanisms. Weights, results, and visual results will be provided as the repository develops. This repository contains implementations, training scripts, weights,and visual results for different image classification models in PyTorch. 

If you have any suggestions, feel free to email me: richardgtang@gmail.com

Note: All pretrained models are done on ImageNet100 due to resource constraints with stored default configs. Download link of ImageNet100 dataset can be found here.

## Results
| Model         | ImageNet (Accuracy/Precision/Recall)          | CIFAR-10 (Accuracy/Precision/Recall)          | MNIST (Accuracy/Precision/Recall)           |
|---------------|:---------------------------------------------:|:---------------------------------------------:|:-------------------------------------------:|
| ResNet18      |                78.9%/79.1%/78.9%               |                92.6%/92.6%/92.6%               |                                             |
| ResNet34      |                80.1%/80.3%/80.1%               |                94.5%/94.5%/94.5%               |                                             |
| MobileNet V.1 |                72.1%/72.3%/72.1%               |                92.2%/92.2%/92.2%               |                                             |
