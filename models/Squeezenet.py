import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, input_channels : int, squeeze_channels : int, expand_channels : int):
        super().__init__()
        self.squeeze = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.squeeze(x))
        return torch.cat([
            self.relu(self.expand1x1(x)),
            self.relu(self.expand3x3(x))
        ], 1)

class SqueezeNetV1(nn.Module):
    def __init__(self, input_channels : int, num_classes : int):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.fire2 = FireModule(96, 16, 64)
        self.fire3 = FireModule(128, 16, 64)
        self.fire4 = FireModule(128, 32, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire5 = FireModule(256, 32, 128)
        self.fire6 = FireModule(256, 48, 192)
        self.fire7 = FireModule(384, 48, 192)
        self.fire8 = FireModule(384, 64, 256)
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire9 = FireModule(512, 64, 256)
        
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
class SqueezeNetv2(nn.Module): 
    def __init__(self, input_channels : int, num_classes : int): 
        super().__init__() 

        self.inital_conv = nn.Sequential(*[nn.Conv2d(input_channels, 96, kernel_size=7, stride=2, padding=2),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        
        self.fire2 = FireModule(96, 16, 64)
        self.fire3 = FireModule(128, 16, 64)
        self.fire4 = FireModule(128, 32, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire5 = FireModule(256, 32, 128)
        self.fire6 = FireModule(256, 48, 192)
        self.fire7 = FireModule(384, 48, 192)
        self.fire8 = FireModule(384, 64, 256)
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire9 = FireModule(512, 64, 256)
        
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x): 
        x = self.inital_conv(x)
        x = self.fire2(x)
        x = x + self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = x + self.fire5(x)
        x = self.fire6(x)
        x = x + self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = x + self.fire9(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
class SqueezeNetv3(nn.Module): 
    def __init__(self, input_channels : int, num_classes : int): 
        super().__init__() 

        self.inital_conv = nn.Sequential(*[nn.Conv2d(input_channels, 96, kernel_size=7, stride=2, padding=2),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
                
        self.conv_1x1_icf1 = nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0)
        self.fire2 = FireModule(96, 16, 64)
        self.fire3 = FireModule(128, 16, 64)
        self.conv_1x1_f3f4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.fire4 = FireModule(128, 32, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire5 = FireModule(256, 32, 128)
        self.conv_1x1_f5f6 = nn.Conv2d(256, 384, kernel_size=1, stride=1, padding=0)
        self.fire6 = FireModule(256, 48, 192)
        self.fire7 = FireModule(384, 48, 192)
        self.conv_1x1_f7f8 = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.fire8 = FireModule(384, 64, 256)
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire9 = FireModule(512, 64, 256)
        
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x): 
        x = self.inital_conv(x)
        x = self.fire2(x) + self.conv_1x1_icf1(x)
        x = x + self.fire3(x) 
        x = self.fire4(x) + self.conv_1x1_f3f4(x)
        x = self.maxpool4(x)
        x = x + self.fire5(x)
        x = self.fire6(x) + self.conv_1x1_f5f6(x)
        x = x + self.fire7(x)
        x = self.fire8(x) + self.conv_1x1_f7f8(x)
        x = self.maxpool8(x)
        x = x + self.fire9(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
def get_SqueezenetV1(input_channels : int = 3, num_classes : int = 10):
    """
    Helper Function for defining SqueezenetV1 with no bypass
    """
    return SqueezeNetV1(input_channels = input_channels, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_SqueezenetV2(input_channels : int = 3, num_classes : int = 10): 
    """
    Helper Function for defining SqueezenetV2 with simple bypass
    """
    return SqueezeNetv2(input_channels=input_channels, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_SqueezenetV3(input_channels : int = 3, num_classes : int = 10):
    """
    Helper Function for defining SqueezenetV3 with complex bypass
    """
    return SqueezeNetv3(input_channels=input_channels, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")