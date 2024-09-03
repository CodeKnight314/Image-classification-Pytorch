import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = torch.nn.functional.relu(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv_stack_1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )

        self.conv_stack_2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )

        self.conv_stack_3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )

        self.conv_stack_4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.conv_stack_5 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adppool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.maxpool(x)
        x = self.conv_stack_2(x)
        x = self.maxpool(x)
        x = self.conv_stack_3(x)
        x = self.maxpool(x)
        x = self.conv_stack_4(x)
        x = self.maxpool(x)
        x = self.conv_stack_5(x)
        x = self.maxpool(x)

        x = self.adppool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv_stack_1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )

        self.conv_stack_2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )

        self.conv_stack_3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )

        self.conv_stack_4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.conv_stack_5 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adppool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.maxpool(x)
        x = self.conv_stack_2(x)
        x = self.maxpool(x)
        x = self.conv_stack_3(x)
        x = self.maxpool(x)
        x = self.conv_stack_4(x)
        x = self.maxpool(x)
        x = self.conv_stack_5(x)
        x = self.maxpool(x)

        x = self.adppool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def get_VGG16(num_classes : int): 
    """
    """
    return VGG16(num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_VGG19(num_classes : int): 
    """
    """
    return VGG19(num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
