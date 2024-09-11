# Image-classification-Pytorch

## Overview
This repository is built for educational purposes on image classification using various computer vision models. The main goal is to demonstrate how these models can be applied to classify images and perform compared to contemporary models on widely known datasets. Much of the novel mechanisms developed and published by others (hopefully my own in the future as well) are reimplemented on this repository to demonstrate the intuition and advantages behind such mechanisms. Weights, results, and visual results will be provided as the repository develops. This repository contains implementations, training scripts, weights,and visual results for different image classification models in PyTorch. 

All trained and available model weights can be found [here](https://drive.google.com/drive/folders/1ByxKdMnFL4uu3fHuKiT-EXcM_HendMgh?usp=sharing)

If you have any suggestions, feel free to email me: richardgtang@gmail.com

Note: All pretrained models are done on ImageNet100 due to resource constraints with stored default configs. Download link of ImageNet100 dataset can be found on [kaggle](https://www.kaggle.com/datasets/wilyzh/imagenet100)

## Results
| Model         | ImageNet100 (Accuracy/Precision/Recall)       | Param Size        |
|---------------|:---------------------------------------------:|:-----------------:|
| ResNet18      |                78.9%/79.1%/78.9%              |       11.7M       |
| ResNet34      |                80.1%/80.3%/80.1%              |       21.8M       |
| MobileNet V.1 |                72.1%/72.3%/72.1%              |       4.3M        |
| SqueezeNet V.3|                74.6%/74.9%/74.5%              |       1.6M        |
| InceptionNetV3|                82.4%/82.6%/82.5%              |       27.0M       |
| VGG16         |                84.2%/84.8%/84.2%              |       37.7M       |
| VGG19         |                83.9%/84.5%/83.9%              |       43.0M       |
| EfficientNetV2|                76.0%/77.0%/76.0%              |       3.4M        |
| DenseNet121   |                75.1%/76.4%/75.1%              |       8.0M        |
| DenseNet169   |                75.4%/78.4%/75.4%              |       14.1M       |
| DenseNet201   |                80.8%/81.6%/80.9%              |       20.2M       |
| CaiT          |                      N/A                      |        N/A        |
| ConvMixer     |                      N/A                      |        N/A        |
| CoAtNet       |                      N/A                      |        N/A        |
| CvT           |                      N/A                      |        N/A        |

## Special note
- While most models in this repository can be directly trained on image classification datasets, pretraining even on a subset of Imagenet yields significant benefit before proper training.
- I've given up on training transformer-based models for now since their inherent need for large-data exceeds my current circumstance (unless I want my laptop to run continuously for 300+ hours). This situation may likely change once I've gained permission to use institutioal resources for free. Transformer-based models are still added for educational purposes.
- Curiously, SqueezeNet V.3 performs poorly on CIFAR-10 compared to other models, most likely from loss of spatial features from already limited spatial information after several pooling operations. However, this is not replicated on MNIST likely due to the simplicity of MNIST training samples.
## Usage
- For training selected models, run the following after cloning the github repository:
```python
python main.py --model MODEL --root_dir ROOT_DIR --config_file CONFIG_FILE
```

- For evaluating selected models, run the following after cloning the github repository:
```python
python evaluate.py --model MODEL --model_save_path MODEL_PATH --root_dir ROOT_DIR --config_file CONFIG_FILE --output_dir OUTPUT_DIR
```

- To check arguement descriptions, run selected scripts with the `--help` flag.
```python
python evaluate.py --help
```
```python
python main.py --help
```
