import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Copied from pytorch-DDSP
    Implementation of the MLP, as described in the original paper

    Parameters :
        in_size (int)   : input size of the MLP
        out_size (int)  : output size of the MLP
        loop (int)      : number of repetition of Linear-Norm-ReLU
    """
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            )] + [nn.Sequential(nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ) for i in range(loop - 1)])

    def forward(self, x):
        for lin in self.linear:
            x = lin(x)
        return x

class FiLM(nn.Module):
    """
    feature-wise linear modulation
    """
    def __init__(self, input_dim, attribute_dim):
        super().__init__()
        self.input_dim = input_dim
        self.generator = nn.Linear(attribute_dim, input_dim*2)
        
    def forward(self, x, c):
        """
        x: (*, input_dim)
        c: (*, attribute_dim)
        """
        c = self.generator(c)
        gamma = c[..., :self.input_dim]
        beta = c[..., self.input_dim:]
        return x*gamma + beta

class FiLMMLP(nn.Module):
    """
    MLP with FiLMs in between
    """
    def __init__(self, in_size, out_size, attribute_dim, loop=3):
        super().__init__()
        self.loop = loop
        self.mlps = nn.ModuleList([nn.Linear(in_size, out_size)] 
                                + [nn.Linear(out_size, out_size) for i in range(loop-1)])
        self.films = nn.ModuleList([FiLM(out_size, attribute_dim) for i in range(loop)])

    def forward(self, x, c):
        """
        x: (*, input_dim)
        c: (*, attribute_dim)
        """
        for i in range(self.loop):
            x = self.mlps[i](x)
            x = F.relu(x)
            x = self.films[i](x, c)
        return x

class Normalize1d(nn.Module):
    """
    normalize over the last dimension
    ddsp normalizes over time dimension of mfcc
    """
    def __init__(self, channels, norm_type='instance', batch_dims=1):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(channels, affine=True)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(channels, affine=True)
        self.flat = nn.Flatten(0, batch_dims-1)

    def forward(self, x):
        """
        First b_dim dimensions are batch dimensions
        Last dim is normalized
        """
        orig_shape = x.shape
        x = self.flat(x)
        if len(x.shape) == 2:
            # no channel dimension
            x = x.unsqueeze(1)
        x = self.norm(x)
        x = x.view(orig_shape)
        return x

class Normalize2d(nn.Module):
    """
    take the average over 2 dimensions (time, frequency)
    """
    def __init__(self, norm_type='instance'):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(1)
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(1, affine=False)

    def forward(self, x):
        """
        3D input first of which is batch dim
        [batch, dim1, dim2]
        """
        x = self.norm(x.unsqueeze(1)).squeeze(1) # dummy channel
        return x

class CoordConv1D(nn.Module):
    # input dimension needs to be fixed
    def __init__(self, in_channels, out_channels, input_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        # 0~1
        pos_embed = torch.arange(input_dim, dtype=torch.float)[None, None, :] / (input_dim)
        # -1~1
        pos_embed = pos_embed * 2 -1
        self.input_dim = input_dim
        self.kernel_size = (kernel_size,)
        self.stride = (stride,)
        self.padding = (padding,)
        self.dilation = (dilation,)

        self.register_buffer('pos_embed', pos_embed)
        self.conv = nn.Conv1d(in_channels+1, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        # x: batch, C_in, H
        batch_size, c_in, h = x.shape
        coord = self.pos_embed.expand(batch_size, -1, -1)
        x = torch.cat([x, coord], dim=1)
        x = self.conv(x)
        return x

class Resnet1D(nn.Module):
    """Resnet for encoder/decoder similar to Jukebox
    """
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=3, reverse_dilation=False):
        """init

        Args:
            n_in (int): input channels
            n_depth (int): depth of resnet
            m_conv (float, optional): multiplier for intermediate channel. Defaults to 1.0.
            dilation_growth_rate (int, optional): rate of exponential dilation growth . Defaults to 1.
            reverse_dilation (bool, optional): reverse growing dilation for encoder/decoder symmetry. Defaults to False.
        """
        super().__init__()
        conv_block = lambda input_channels, inner_channels, dilation: nn.Sequential(
            nn.ReLU(),
            # this conv doesn't change size
            nn.Conv1d(input_channels, inner_channels, 3, 1, dilation, dilation),
            nn.ReLU(),
            #1x1 convolution
            nn.Conv1d(inner_channels, input_channels, 1, 1, 0),
        )
        # blocks of convolution with growing dilation
        conv_blocks = [conv_block(n_in, int(m_conv * n_in),
                                dilation=dilation_growth_rate ** depth)
                                for depth in range(n_depth)]
        if reverse_dilation: # decoder should be flipped backwards
            conv_blocks = conv_blocks[::-1]
        self.blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        for block in self.blocks:
            # residual connection
            x = x + block(x)
        return x

class Resnet2D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=3, reverse_dilation=False):
        """init

        Args:
            n_in (int): input channels
            n_depth (int): depth of resnet
            m_conv (float, optional): multiplier for intermediate channel. Defaults to 1.0.
            dilation_growth_rate (int, optional): rate of exponential dilation growth . Defaults to 3.
        """
        super().__init__()
        conv_block = lambda input_channels, inner_channels, dilation: nn.Sequential(
            nn.ReLU(),
            # this conv doesn't change size
            nn.Conv2d(input_channels, inner_channels, 3, 1, dilation, dilation),
            nn.ReLU(),
            #1x1 convolution
            nn.Conv2d(inner_channels, input_channels, 1, 1, 0),
        )
        # blocks of convolution with growing dilation
        conv_blocks = [conv_block(n_in, int(m_conv * n_in),
                                dilation=dilation_growth_rate ** depth)
                                for depth in range(n_depth)]
        if reverse_dilation: # decoder should be flipped backwards
            conv_blocks = conv_blocks[::-1]
        self.blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        for block in self.blocks:
            # residual connection
            x = x + block(x)
        return x