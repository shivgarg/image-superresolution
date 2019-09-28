import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import PIL



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
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        # Residual blocks
        self.res_blocks1= ResidualBlock(128)

        # Upsampling layers
        self.upsampling0 = nn.Sequential(nn.Upsample(scale_factor=2),
                                          nn.Conv2d(128,64,3,1,1),
 	                                         nn.ReLU())
	# Residual blocks        
        self.res_block3 = ResidualBlock(64)    
    
        self.upsampling1 = nn.Sequential(nn.Upsample(scale_factor=2),
                                          nn.Conv2d(64,64,3,1,1),
                                          nn.ReLU())
	# Residual blocks
        self.res_block4 = ResidualBlock(64)    

        # Final output layer
        self.conv7 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())

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
        out = self.conv7(out1)
        return out




model = GeneratorResNet()
model.eval()
model.load_state_dict(torch.load('project/6/generator_4.pth'))    
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def superresolve(img, seed=2019):
    """
    YOUR CODE HERE
    
    Superresolve an image by a factor of 4
    @img: A PIL image
    @return A PIL image
    
    e.g:
    >>> img.size
    >>> (64,64)
    >>> img = superresolve(img)
    >>> img.size
    >>> (256,256)
    """
    torch.random.manual_seed(seed)
    img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )
    x = img_transform(img)
    x = x.unsqueeze(0)
    noise = torch.randn(x.size())*0.01
    x += noise
    x = torch.squeeze(model(x))
    img_output = transforms.Compose(
                    [
                        #transforms.Normalize(-1*(mean/std),1./std),
                        transforms.ToPILImage(mode='RGB')
                    ]
            )
    im = img_output(x)
    return im
    


