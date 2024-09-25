import importlib

MODEL_MAPPING = {
    'ViT': 'ViT.get_ViT',
    'ResNet18': 'ResNet.get_ResNet18',
    'ResNet34': 'ResNet.get_ResNet34',
    'CvT-13': 'CvT.get_CVT13',
    'CvT-21': 'CvT.get_CVT21',
    'CvT-24': 'CvT.get_CVTW24',
    'MobileNet': 'MobileNet.get_MobileNet',
    'Squeezenetv3': 'Squeezenet.get_SqueezenetV3',
    'InceptionNetv3': 'InceptionNet.get_InceptionNetV3',
    'VGG16': 'VGG.get_VGG16',
    'VGG19': 'VGG.get_VGG19',
    'DenseNet121': 'DenseNet.get_DenseNet121',
    'DenseNet169': 'DenseNet.get_DenseNet169',
    'DenseNet201': 'DenseNet.get_DenseNet201',
    'DenseNet264': 'DenseNet.get_DenseNet264',
    'EfficientNetV2': 'EfficientNet.get_EfficientNetV2',
    'ConvMixer': 'ConvMixer.get_ConvMixer'
}

def load_model_class(model_name):
    if model_name in MODEL_MAPPING:
        module_class = MODEL_MAPPING[model_name]
        module_name, class_name = module_class.rsplit('.', 1)
        module = importlib.import_module(f'models.{module_name}')
        model_class = getattr(module, class_name)
        return model_class
    else:
        raise ValueError(f"Model {model_name} is not defined in the model mapping.")
