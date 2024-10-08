import torch 
import torch.nn as nn 
import optuna 
from dataset import * 
from tqdm import tqdm 

class ResBlock(nn.Module): 
    def __init__(self, input_channels: int, output_channels: int, stride: int): 
        super().__init__()

        if input_channels != output_channels: 
            self.projection = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_channels)
            )
        else: 
            self.projection = nn.Identity()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.ba_n1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.ba_n2 = nn.BatchNorm2d(output_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        
        out = self.relu(self.ba_n1(self.conv1(x)))
        out = self.ba_n2(self.conv2(out))
        out = self.relu(out)  
        
        out = out + identity
        out = self.relu(out) 
    
        return out

class ResStack(nn.Module): 
    def __init__(self, input_channels : int, output_channels : int, num_layers : int, stride : int = 1): 
        super().__init__() 

        layers = []
        layers.append(ResBlock(input_channels=input_channels, output_channels=output_channels, stride=stride))
        for _ in range(num_layers - 1):
            layers.append(ResBlock(input_channels=output_channels, output_channels=output_channels, stride=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        return self.block(x)

class ResNet(nn.Module): 
    def __init__(self, channels = [64, 128, 256, 512], num_layers = [3, 4, 6, 3], num_classes : int = 10):
        super().__init__()

        assert len(channels) == len(num_layers), "[ERROR] Channels and Layers lists do not match in length."

        self.channels = channels
        self.num_layers = num_layers

        self.input_conv = nn.Sequential(*[nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64), 
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=3, stride=2)])
        
        blocks = []
        blocks.append(ResStack(64, channels[0],num_layers[0], 1))
        for i in range(1, len(channels)):
            blocks.append(ResStack(channels[i-1], channels[i], num_layers[i], 2))

        self.blocks = nn.Sequential(*blocks)

        self.classifier_head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                            nn.Flatten(), 
                                            nn.Linear(channels[-1], num_classes), 
                                            nn.Dropout(0.1))

    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        first_conv = self.input_conv(x)
        block_conv = self.blocks(first_conv)
        logits = self.classifier_head(block_conv)

        return logits
    
def get_ResNet18(num_classes: int = 10): 
    """
    Helper function for getting ResNet18

    Returns:
        ResNet: ResNet model with four layers with two blocks each at 64, 128, 256, 512 channels, respectively.
    """
    return ResNet(channels=[64, 128, 256, 512], num_layers=[2, 2, 2, 2], num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

def get_ResNet34(num_classes: int = 10):
    """
    Helper function for getting ResNet34

    Returns: 
        ResNet: ResNet model with four layers with 3, 4, 6, 3 blocks at 64, 128, 256, 512 channels, respectively.
    """
    return ResNet(channels=[64, 128, 256, 512], num_layers=[3, 4, 6, 3], num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")