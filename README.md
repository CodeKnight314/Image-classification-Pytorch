# Image-classification-Pytorch

## Overview
This repository is built for educational purposes on image classification using various computer vision models. The main goal is to demonstrate how these models can be applied to classify images and perform compared to contemporary models on widely known datasets. Much of the novel mechanisms developed and published by others (hopefully my own in the future as well) are reimplemented on this repository to demonstrate the intuition and advantages behind such mechanisms. Weights, results, and visual results will be provided as the repository develops. This repository contains implementations, training scripts, weights,and visual results for different image classification models in PyTorch. 

If you have any suggestions, feel free to email me: richardgtang@gmail.com

Note: All pretrained models are done on ImageNet100 due to resource constraints with stored default configs. Download link of ImageNet100 dataset can be found here.

## Results
| Model         | CIFAR-10 Accuracy | CIFAR-10 Precision | CIFAR-10 Recall | ImageNet Accuracy | ImageNet Precision | ImageNet Recall | MNIST Accuracy | MNIST Precision | MNIST Recall |
|---------------|--------------------|--------------------|-----------------|-------------------|--------------------|-----------------|----------------|-----------------|--------------|
| ResNet18      |      0.925881      |      0.926343      |     0.925856    |     0.788662      |      0.791496      |     0.788738    |                |                 |              |
| ResNet34      |      0.945012      |      0.945158      |     0.945009    |     0.800681      |      0.802748      |     0.800717    |                |                 |              |
| MobileNet V.1 |      0.921875      |      0.922063      |     0.921876    |     0.720954      |      0.722931      |     0.720955    |                |                 |              |
