import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from files.octConv import MultiOctaveConv


class ResidualBlock(nn.Module):
    def safe_sum(self,a,b):
        if type(a) is tuple:
            output = []
            for i in range(len(a)):
                if a[i] is None:
                    output.append(b[i])
                elif b[i] is None:
                    output.append(a[i])
                else:
                    output.append(a[i]+b[i])
            return tuple(output)
        else:
            return a + b[0]
        
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, alpha, beta, last=False,i=0):
        super(ResidualBlock, self).__init__()
        self.i = i
        if last:
            self.up_layer = MultiOctaveConv(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias = False,
                                            alpha_in=alpha, alpha_out=0.0, beta_in=beta, beta_out=0.0,
                                            norm_layer = torch.nn.InstanceNorm2d)
        else:
            self.up_layer = None
        layers = []
        layers.append(MultiOctaveConv(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias = False,
                                      alpha_in=alpha, alpha_out=alpha, beta_in=beta, beta_out=beta,
                                      activation_layer=nn.ReLU(inplace=True), norm_layer = torch.nn.InstanceNorm2d)
        )
        if last:
            layers.append(MultiOctaveConv(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias = False,
                                          alpha_in=alpha, alpha_out=0.0, beta_in=beta, beta_out=0.0,
                                          activation_layer=nn.ReLU(inplace=True), norm_layer = torch.nn.InstanceNorm2d)
            )
        else:
            layers.append(MultiOctaveConv(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias = False,
                                          alpha_in=alpha, alpha_out=alpha, beta_in=beta, beta_out=beta,
                                          norm_layer = torch.nn.InstanceNorm2d)
            )
        self.main = nn.Sequential(*layers)
        #)
    def forward(self, x):
        if self.up_layer is None:
            return self.safe_sum(x, self.main(x))
        else:
            return self.up_layer(x)[0] + self.main(x)[0]

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, alpha=0.5, beta=0.0):
        super(Generator, self).__init__()

        layers = []
        layers.append(MultiOctaveConv(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False,
                                          alpha_in=0, alpha_out=alpha, beta_in=0, beta_out=0,
                                          activation_layer=nn.ReLU(inplace=True), norm_layer = torch.nn.InstanceNorm2d)
        )
        curr_dim = conv_dim
        layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False,
                                          alpha_in=alpha, alpha_out=alpha, beta_in=0, beta_out=beta,
                                          activation_layer=nn.ReLU(inplace=True), norm_layer = torch.nn.InstanceNorm2d)
                     )
        curr_dim = curr_dim * 2
        
        layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False,
                                          alpha_in=alpha, alpha_out=alpha, beta_in=beta, beta_out=beta,
                                          activation_layer=nn.ReLU(inplace=True), norm_layer = torch.nn.InstanceNorm2d)
                     )
        curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(1,repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, alpha=alpha, beta=beta, i=i))
        layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, alpha=alpha, beta=beta, last=True, i=i+1))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, kernel_size=4, alpha=0.5, beta=0.):
        super(Discriminator, self).__init__()
        layers = []
        padding = 1
    
        if repeat_num == 1:
            alpha = 0
            beta = 0
        alpha_in = alpha
        alpha_out = alpha
        beta_in = beta
        beta_out = beta
        
        curr_dim = conv_dim
        flag_1 = False
        flag_2 = False
        size_in = image_size//2
        layers.append(MultiOctaveConv(3, conv_dim, kernel_size=kernel_size, stride=2, padding=padding,
                                          alpha_in=0, alpha_out=alpha, beta_in=0, beta_out=0,
                                          activation_layer=nn.LeakyReLU(0.01)) )
        
        layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=kernel_size, stride=2, padding=padding,
                                          alpha_in= alpha_in, alpha_out=alpha_out, beta_in=0, beta_out=beta_out,
                                          activation_layer=nn.LeakyReLU(0.01)))
        curr_dim = curr_dim * 2
        size_in //= 2
            
        for i in range(2, repeat_num):
            layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=kernel_size, stride=2, padding=padding,
                                          alpha_in= alpha_in, alpha_out=alpha_out, beta_in=beta_in, beta_out=beta_out,
                                          activation_layer=nn.LeakyReLU(0.01)))
            curr_dim = curr_dim * 2
            size_in //= 2
            
            if not flag_1 and size_in < 16:
                beta_out = 0
                alpha_out += beta_out
                flag_1 = True
                continue
            if flag_1:
                beta_in = 0
                
            if not flag_2 and size_in < 8:
                alpha_out = 0
                flag_2 = True
                continue
            if flag_2:
                alpha_in = 0
            
            
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = MultiOctaveConv(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False,
                                     alpha_in=alpha_in, beta_in=beta_in, beta_out=0, alpha_out=0)
        self.conv2 = MultiOctaveConv(curr_dim, c_dim, kernel_size=kernel_size, bias=False, padding=0,
                                     alpha_in=alpha_in, beta_in=beta_in, beta_out=0, alpha_out=0, last=True)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)[0]
        out_cls = self.conv2(h)[0]
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
