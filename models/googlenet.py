import torch 
import torch.nn as nn 

class Stem(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__() 

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=2, padding=0)
        self.conv6 = nn.Conv2d(192, 288, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x): 
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        x = self.maxpool(x)

        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))

        return x 
    
class InceptionModuleA(nn.Module): 
    def __init__(self, input_channels):
        super().__init__() 

        self.conv_branch_1 = nn.Sequential(*[nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0)])

        self.conv_branch_2 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                             nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0)])
        
        self.conv_branch_3 = nn.Sequential(*[nn.Conv2d(input_channels, 48, kernel_size=1, stride=1, padding=0), 
                                             nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)])
        
        self.conv_branch_4 = nn.Sequential(*[nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0), 
                                             nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1), 
                                             nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)])
        
    def forward(self, x): 
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv3 = self.conv_branch_3(x)
        conv4 = self.conv_branch_4(x)

        output = torch.concat([conv1, conv2, conv3, conv4], dim=1)

        return output
    
class InceptionModuleB(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__() 

        self.conv_branch_1 = nn.Conv2d(input_channels, 192, kernel_size=1, stride=1, padding=0)

        self.conv_branch_2 = nn.Sequential(*[nn.Conv2d(input_channels, 128, kernel_size=1, stride=1, padding=0), 
                                             nn.Conv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)), 
                                             nn.Conv2d(128, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))])
        
        self.conv_branch_3 = nn.Sequential(*[nn.Conv2d(input_channels, 160, kernel_size=1, stride=1, padding=0), 
                                             nn.Conv2d(160, 160, kernel_size=(7, 1), stride=1, padding=(3, 0)), 
                                             nn.Conv2d(160, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), 
                                             nn.Conv2d(160, 160, kernel_size=(7, 1), stride=1, padding=(3, 0)), 
                                             nn.Conv2d(160, 192, kernel_size=(1, 7), stride=1, padding=(0, 3))])
        
        self.conv_branch_4 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                                             nn.Conv2d(input_channels, 192, kernel_size=1, stride=1, padding=0)])
        
    def forward(self, x): 
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv3 = self.conv_branch_3(x)
        conv4 = self.conv_branch_4(x)

        output = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        return output
    
class InceptionModuleC(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__()

        self.conv_branch_1 = nn.Conv2d(input_channels, 320, kernel_size=1, stride=1, padding=0)

        self.conv_branch_2 = nn.Conv2d(input_channels, 384, kernel_size=1, stride=1, padding=0)
        self.conv_branch_2_a = nn.Conv2d(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_branch_2_b = nn.Conv2d(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_branch_3 = nn.Sequential(*[nn.Conv2d(input_channels, 448, kernel_size=1, stride=1, padding=0), 
                                             nn.Conv2d(448, 384, kernel_size=3, stride=1, padding=1)])
        
        self.conv_branch_3_a = nn.Conv2d(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_branch_3_b = nn.Conv2d(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_branch_4 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                             nn.Conv2d(input_channels, 192, kernel_size=1, stride=1, padding=0)])
        
    def forward(self, x): 
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv2a = self.conv_branch_2_a(conv2)
        conv2b = self.conv_branch_2_b(conv2)
        conv3 = self.conv_branch_3(x)
        conv3a = self.conv_branch_3_a(conv3)
        conv3b = self.conv_branch_3_b(conv3)
        conv4 = self.conv_branch_4(x)

        output = torch.cat([conv1, conv2a, conv2b, conv3a, conv3b, conv4], dim=1)

        return output