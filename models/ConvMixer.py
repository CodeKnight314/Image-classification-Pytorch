import torch 
import torch.nn as nn 

class PatchEmbeddingConv(nn.Module): 
    def __init__(self, input_channels : int = 3, patch_size : int = 8, output_channels : int = 64): 
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=patch_size, stride=patch_size, padding=0)
        self.non_linear = nn.GELU() 
        self.batch_norm = nn.BatchNorm2d(output_channels)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        x = self.conv(x)
        x = self.non_linear(x)
        x = self.batch_norm(x)
        return x
    
class ConvMixerModule(nn.Module): 
    def __init__(self, input_channels : int, output_channels : int, kernel_size : int = 3):
        super().__init__() 
        
        self.depth_conv = nn.Sequential(*[
            nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=1, padding=1, groups=input_channels),
            nn.GELU(), 
            nn.BatchNorm2d(input_channels)])

        self.point_conv = nn.Sequential(*[
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0), 
            nn.GELU(),
            nn.BatchNorm2d(output_channels)
        ])
        
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        residual = x
        x = self.depth_conv(x) + residual
        x = self.point_conv(x)
        
        return x

class ConvMixer(nn.Module): 
    def __init__(self, input_channels, dim, kernel_size, patch_size, depth, num_classes):
        super().__init__()
        
        self.patch_embed = PatchEmbeddingConv(input_channels=input_channels, patch_size=patch_size, output_channels=dim)
        
        self.convMixerStem = nn.Sequential(*[ConvMixerModule(dim, dim, kernel_size) for _ in range(depth)])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifer = nn.Linear(512, num_classes)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        x = self.patch_embed(x)
        x = self.convMixerStem(x)
        x = self.flatten(self.avg_pool(x))
        output = self.classifer(x)
        return output
    
    
def get_ConvMixer(num_classes : int = 10): 
    return ConvMixer(input_channels=3, dim=128, kernel_size=9, patch_size=7, depth=4, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
    