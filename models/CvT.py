import torch 
import torch.nn as nn
from typing import Tuple
import math

class MHSA_Conv(nn.Module): 
    def __init__(self, dim : int, head : int): 
        super().__init__()

        self.d_k = dim // head
        self.head = head

        self.qkv_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.q_pconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.k_pconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.v_pconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(0.1)

    def scaled_dot_product(self, Queries, Keys, Values):
        attn_score = torch.matmul(Queries, torch.transpose(Keys, -2, -1)) / math.sqrt(self.d_k)
        QK_probs = torch.softmax(attn_score, dim = -1)
        QK_probs = self.dropout(QK_probs)
        output = torch.matmul(QK_probs, Values)
        return output

    def forward(self, x): 
        B, C, H, W = x.shape 

        proj = self.qkv_conv(x)
        
        Q = self.q_pconv(proj).view(B, self.head, -1, self.d_k).transpose(2, 3)
        K = self.k_pconv(proj).view(B, self.head, -1, self.d_k).transpose(2, 3)
        V = self.v_pconv(proj).view(B, self.head, -1, self.d_k).transpose(2, 3)

        context = self.scaled_dot_product(Q, K, V).view(B, -1, self.d_k * self.head)

        return context

class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()

        self.l_1 = nn.Linear(input_dim, hidden_dim)
        self.l_2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the feed-forward network. Applies a linear transformation followed by a GELU activation,
        a dropout layer, and another linear transformation followed by GELU activation and dropout.

        Args:
            x (torch.Tensor): Input tensor of shape [batch size, feature dimension].

        Returns:
            torch.Tensor: Output tensor of shape [batch size, output dimension].
        """
        x = self.l_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.l_2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x
    
class ConvTransformerBlock(nn.Module): 
    def __init__(self, dim : int, head : int, expansion_factor : int):
        super().__init__() 

        self.msa_conv = MHSA_Conv(dim, head)
        self.msa_norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dim * expansion_factor, dim, dropout_rate=0.1)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x): 
        B, C, H, W = x.shape
        x = x + self.msa_norm(self.msa_conv(x))
        x = x + self.ffn_norm(self.ffn(x))
        x = x.reshape(B, C, H, W)
        return x
    
class ConvTransformerBlock(nn.Module): 
    def __init__(self, dim : int, head : int, expansion_factor : int):
        super().__init__() 

        self.msa_conv = MHSA_Conv(dim, head)
        self.msa_norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dim * expansion_factor, dim, dropout_rate=0.1)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x): 
        B, C, H, W = x.shape
        x = x + self.msa_norm(self.msa_conv(x)).reshape(B, C, H, W)
        x = x + self.ffn_norm(self.ffn(x.reshape(B, -1, C))).reshape(B, C, H, W)
        x = x.reshape(B, C, H, W)
        return x

class CvT(nn.Module): 
    def __init__(self, input_channels : int, channels : Tuple[int], head : Tuple[int], num_layers : Tuple[int], num_classes: int):
        super().__init__()

        self.conv1_embed = nn.Conv2d(input_channels, channels[0], kernel_size=7, stride=4, padding=2)
        self.conv2_embed = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        self.conv3_embed = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        self.t_conv_1 = nn.Sequential(*[ConvTransformerBlock(channels[0], head[0], expansion_factor=4) for _ in range(num_layers[0])])
        self.t_conv_2 = nn.Sequential(*[ConvTransformerBlock(channels[1], head[1], expansion_factor=4) for _ in range(num_layers[1])])
        self.t_conv_3 = nn.Sequential(*[ConvTransformerBlock(channels[2], head[2], expansion_factor=4) for _ in range(num_layers[2])])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], num_classes)

    def forward(self, x): 
        x = self.conv1_embed(x)
        x = self.t_conv_1(x)
        
        x = self.conv2_embed(x)
        x = self.t_conv_2(x)
        
        x = self.conv3_embed(x)
        x = self.t_conv_3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def get_CVT13(num_classes : int, device : str = "cuda"): 
    return CvT(3, channels=[64, 192, 384], head=[1, 3, 6], num_layers=[1, 2, 10], num_classes=num_classes).to(device)

def get_CVT21(num_classes : int, device : str = "cuda"): 
    return CvT(3, channels=[64, 192, 384], head=[1, 3, 6], num_layers=[1, 4, 16], num_classes=num_classes).to(device)

def get_CVTW24(num_classes : int, device : str = "cuda"): 
    return CvT(3, channels=[192, 768, 1024], head=[3, 12, 16], num_layers=[2, 2, 20], num_classes=num_classes).to(device)
