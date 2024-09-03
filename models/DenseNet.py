import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import List

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.avg_pool(out)
        return out
    
class DenseLayer(nn.Module): 
    def __init__(self, in_channels : int, bn_size : int, growth_rate : int, dropout : float): 
        super().__init__() 

        self.conv_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, padding=0)
        )

        self.conv_2 = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        x = torch.cat(x, 1)

        bottleneck_output = self.conv_1(x)
        output = self.dropout(self.conv_2(bottleneck_output))

        return output

class DenseBlock(nn.Module): 
    def __init__(self, in_channels : int, num_layers : int, bn_size : int, growth_rate : int, dropout : float): 
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers): 
            block = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size, 
                dropout=dropout
            )

            self.layers.append(block)

    def forward(self, x): 
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module): 
    def __init__(self, num_initial_features : int, num_blocks : List[int], growth_rate : int, bn_size : int, num_classes : int):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, num_initial_features, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(num_initial_features), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()
        num_features = num_initial_features
        for i in range(len(num_blocks)):
            block = DenseBlock(
                in_channels=num_features, 
                num_layers=num_blocks[i],
                bn_size=bn_size, 
                growth_rate=growth_rate,
                dropout=0.2
            )

            num_features = num_features + num_blocks[i] * growth_rate

            self.blocks.append(block)

            if i != len(num_blocks) - 1: 
                transition = TransitionLayer(num_features, num_features // 2)
                num_features = num_features // 2
                self.blocks.append(transition)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes), 
            nn.Dropout(0.2),
            nn.Softmax()
        )
    
    def forward(self, x): 
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=(1,1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def get_DenseNet121(num_classes: int):
    """
    Helper function for defining DenseNet121
    """
    return DenseNet(num_initial_features=64, num_blocks=[6, 12, 24, 16], growth_rate=32, bn_size=4, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_DenseNet169(num_classes: int):
    """
    Helper function for defining DenseNet169
    """
    return DenseNet(num_initial_features=64, num_blocks=[6, 12, 32, 32], growth_rate=32, bn_size=4, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_DenseNet201(num_classes: int):
    """
    Helper function for defining DenseNet201
    """
    return DenseNet(num_initial_features=64, num_blocks=[6, 12, 48, 32], growth_rate=32, bn_size=4, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_DenseNet264(num_classes: int):
    """
    Helper function for defining DenseNet264
    """
    return DenseNet(num_initial_features=64, num_blocks=[6, 12, 64, 48], growth_rate=32, bn_size=4, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
