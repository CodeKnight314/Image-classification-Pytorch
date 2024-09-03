import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = torch.nn.functional.silu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out
    
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)

        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        self.se = SEBlock(expanded_channels, reduction=int(1 / se_ratio))

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))

        x = self.se(x)

        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            x = x + identity
        return x

class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(FusedMBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size // 2, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)

        self.se = SEBlock(expanded_channels, reduction=int(1 / se_ratio))

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = F.relu6(self.bn0(self.expand_conv(x)))

        x = self.se(x)

        x = self.bn1(self.project_conv(x))

        if self.use_residual:
            x = x + identity
        return x

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetV2, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )

        self.blocks = nn.Sequential(
            FusedMBConvBlock(24, 24, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.25),
            FusedMBConvBlock(24, 48, expand_ratio=4, kernel_size=3, stride=2, se_ratio=0.25),
            FusedMBConvBlock(48, 48, expand_ratio=4, kernel_size=3, stride=1, se_ratio=0.25),
            MBConvBlock(48, 64, expand_ratio=4, kernel_size=3, stride=2, se_ratio=0.25),
            MBConvBlock(64, 128, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25),
            MBConvBlock(128, 160, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25),
            MBConvBlock(160, 256, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25)
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
def get_EfficientNetV2(num_classes : int = 10): 
    """
    Helper function for defining EfficientNetv2
    """
    return EfficientNetV2(num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")