import torch 
import torch.nn as nn 

class Stem(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__() 

        self.conv1 = ConvBlock(input_channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = ConvBlock(64, 80, kernel_size=3, stride=1, padding=0)
        self.conv5 = ConvBlock(80, 192, kernel_size=3, stride=2, padding=0)
        self.conv6 = ConvBlock(192, 288, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x 
    
class AuxClassifer(nn.Module): 
    def __init__(self, input_channels, num_classes, dropout = 0.3): 
        super().__init__() 

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(*[ConvBlock(input_channels, 512, kernel_size=3, stride=1, padding = 1),
                                    ConvBlock(512, 256, kernel_size=3, stride=1, padding=1),
                                    nn.AdaptiveAvgPool2d(output_size=(1,1))])
        self.classifer = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): 
        x = self.maxpool(x)
        x = self.conv(x)
        x = x.view(x.size(), -1)
        x = self.classifer(x)
        x = self.dropout(x)
        return x
    
class ConvBlock(nn.Module): 
    def __init__(self, input_channels, output_channels, **kwargs): 
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(output_channels)
    
    def forward(self, x): 
        x = self.conv(x)
        x = self.batchnorm(x)
        x = torch.nn.functional.relu(x)
        return x
    
class InceptionModuleA(nn.Module): 
    def __init__(self, input_channels):
        super().__init__() 

        self.conv_branch_1 = nn.Sequential(*[ConvBlock(input_channels, 64, kernel_size=1, stride=1, padding=0)])

        self.conv_branch_2 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                             ConvBlock(input_channels, 64, kernel_size=1, stride=1, padding=0)])
        
        self.conv_branch_3 = nn.Sequential(*[ConvBlock(input_channels, 48, kernel_size=1, stride=1, padding=0), 
                                             ConvBlock(48, 64, kernel_size=3, stride=1, padding=1)])
        
        self.conv_branch_4 = nn.Sequential(*[ConvBlock(input_channels, 64, kernel_size=1, stride=1, padding=0), 
                                             ConvBlock(64, 96, kernel_size=3, stride=1, padding=1), 
                                             ConvBlock(96, 96, kernel_size=3, stride=1, padding=1)])
        
    def forward(self, x): 
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv3 = self.conv_branch_3(x)
        conv4 = self.conv_branch_4(x)

        output = torch.concat([conv1, conv2, conv3, conv4], dim=1)

        return output
    
class InceptionReductionA(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv_branch_1 = ConvBlock(input_channels, 384, kernel_size=3, stride=2, padding=0)
        self.conv_branch_2 = nn.Sequential(
            ConvBlock(input_channels, 192, kernel_size=1, stride=1, padding=0),
            ConvBlock(192, 224, kernel_size=3, stride=1, padding=1),
            ConvBlock(224, 256, kernel_size=3, stride=2, padding=0)
        )
        self.conv_branch_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            ConvBlock(input_channels, 128, kernel_size=1, stride=1, padding=0))
    
    def forward(self, x):
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv3 = self.conv_branch_3(x)
        output = torch.cat([conv1, conv2, conv3], dim=1)
        return output
    
class InceptionModuleB(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__() 

        self.conv_branch_1 = ConvBlock(input_channels, 192, kernel_size=1, stride=1, padding=0)

        self.conv_branch_2 = nn.Sequential(*[ConvBlock(input_channels, 128, kernel_size=1, stride=1, padding=0), 
                                             ConvBlock(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)), 
                                             ConvBlock(128, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))])
        
        self.conv_branch_3 = nn.Sequential(*[ConvBlock(input_channels, 160, kernel_size=1, stride=1, padding=0), 
                                             ConvBlock(160, 160, kernel_size=(7, 1), stride=1, padding=(3, 0)), 
                                             ConvBlock(160, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), 
                                             ConvBlock(160, 160, kernel_size=(7, 1), stride=1, padding=(3, 0)), 
                                             ConvBlock(160, 192, kernel_size=(1, 7), stride=1, padding=(0, 3))])
        
        self.conv_branch_4 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                                             ConvBlock(input_channels, 192, kernel_size=1, stride=1, padding=0)])
        
    def forward(self, x): 
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv3 = self.conv_branch_3(x)
        conv4 = self.conv_branch_4(x)

        output = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        return output
    
class InceptionReductionB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv_branch_1 = nn.Sequential(
            ConvBlock(input_channels, 192, kernel_size=1, stride=1, padding=0),
            ConvBlock(192, 192, kernel_size=3, stride=2, padding=0)
        )
        self.conv_branch_2 = nn.Sequential(
            ConvBlock(input_channels, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(320, 320, kernel_size=3, stride=2, padding=0)
        )
        self.conv_branch_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        conv1 = self.conv_branch_1(x)
        conv2 = self.conv_branch_2(x)
        conv3 = self.conv_branch_3(x)
        output = torch.cat([conv1, conv2, conv3], dim=1)
        return output
    
class InceptionModuleC(nn.Module): 
    def __init__(self, input_channels): 
        super().__init__()

        self.conv_branch_1 = ConvBlock(input_channels, 320, kernel_size=1, stride=1, padding=0)

        self.conv_branch_2 = ConvBlock(input_channels, 384, kernel_size=1, stride=1, padding=0)
        self.conv_branch_2_a = ConvBlock(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_branch_2_b = ConvBlock(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_branch_3 = nn.Sequential(*[ConvBlock(input_channels, 448, kernel_size=1, stride=1, padding=0), 
                                             ConvBlock(448, 384, kernel_size=3, stride=1, padding=1)])
        
        self.conv_branch_3_a = ConvBlock(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_branch_3_b = ConvBlock(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_branch_4 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                             ConvBlock(input_channels, 192, kernel_size=1, stride=1, padding=0)])
        
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
    
class InceptionNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = Stem(3)
        self.inception_a = nn.Sequential(
            InceptionModuleA(288),
            InceptionModuleA(288),
            InceptionModuleA(288),
            InceptionReductionA(288)
        )
        self.inception_b = nn.Sequential(
            InceptionModuleB(768),
            InceptionModuleB(768),
            InceptionModuleB(768),
            InceptionModuleB(768),
            InceptionModuleB(768),
            InceptionReductionB(768)
        )
        self.inception_c = nn.Sequential(
            InceptionModuleC(1280),
            InceptionModuleC(2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.inception_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def get_InceptionNetV3(num_classes : int): 
    return InceptionNetV3(num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")