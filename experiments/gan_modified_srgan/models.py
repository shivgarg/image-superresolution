import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16
import math


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


#class ResidualBlock(nn.Module):
#    def __init__(self, in_features):
#        super(ResidualBlock, self).__init__()
#        self.conv_block = nn.Sequential(
#            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(in_features),
#            nn.PReLU(),
#            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(in_features),
#        )

#    def forward(self, x):
#        return x + self.conv_block(x)

#class GeneratorResNet(nn.Module):
#    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
#        super(GeneratorResNet, self).__init__()
#
#        # First layer
#        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())
#
#        # Residual blocks
#        res_blocks = []
#        for _ in range(4):
#            res_blocks.append(ResidualBlock(64))
#        self.res_blocks = nn.Sequential(*res_blocks)
#
#        # Second conv layer post residual blocks
#        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))
#
#        # Upsampling layers
#        upsampling = []
#        for out_features in range(2):
#            upsampling += [
#                # nn.Upsample(scale_factor=2),
#                nn.Conv2d(64, 64, 3, 1, 1),
#                nn.BatchNorm2d(64),
#                nn.upsample(scake_factor=2),
#                nn.LeakyReLU(),
#            ]
#        self.upsampling = nn.Sequential(*upsampling)
#
#        # Final output layer
#        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4))
#
#    def forward(self, x):
#        out1 = self.conv1(x)
#        out = self.res_blocks(out1)
#        out2 = self.conv2(out)
#        out = torch.add(out1, out2)
#        out = self.upsampling(out)
#        out = self.conv3(out)
#        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU()
        )
        

    def forward(self, x):
        return self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        
        # Residual blocks
        self.res_blocks1= ResidualBlock(64)
        #self.res_blocks2 = ResidualBlock(64)

        # Upsampling layers
        self.upsampling0 = nn.Sequential(nn.Upsample(scale_factor=2),
                                          nn.Conv2d(64,64,3,1,1),
                                          nn.ReLU())
        self.res_block3 = ResidualBlock(64)    
    
        self.upsampling1 = nn.Sequential(nn.Upsample(scale_factor=2),
                                          nn.Conv2d(64,64,3,1,1),
                                          nn.ReLU())
        self.res_block4 = ResidualBlock(64)    
    
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
#        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(64), nn.PReLU())
#        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.PReLU())
#        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(64), nn.PReLU())

        # Final output layer
        self.conv7 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks1(out1)
        out1 = torch.add(out1,out2)
        
        out2 = self.upsampling0(out1)
        out = self.res_block3(out2)
        out1 = torch.add(out, out2)
        out2 = self.upsampling1(out1)
        out = self.res_block4(out2)
        out1 = torch.add(out, out2)
        #out1 = self.conv3(out1)
        out = self.conv7(out1)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 32, 32, 32]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
