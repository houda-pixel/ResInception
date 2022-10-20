import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from collections import OrderedDict

from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))



class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
        out_channels
    ):

        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

        self.out_conv = ConvBlock(out_1x1+out_3x3+out_5x5+out_pool, out_channels, kernel_size=1)
        self.residual_conv = ConvBlock(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        out =  torch.cat([F.relu(branch(x)) for branch in branches], 1)
        return F.relu(self.out_conv(out) + self.residual_conv(x))

@MODELS.register_module()
class ResIncep_FoldingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_pred = config.num_pred
        self.encoder_channel = config.encoder_channel
        self.grid_size = int(pow(self.num_pred,0.5) + 0.5)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )

        # self.folding1 = nn.Sequential(
        #     nn.Conv1d(self.encoder_channel + 2, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 3, 1),
        # )

        self.folding1 = nn.Sequential(

            InceptionBlock(self.encoder_channel + 2, 512, 128, 256, 64, 128, 128, 512),
            InceptionBlock(512, 128, 64, 128, 64, 32, 32, 3)
          
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        # self.folding2 = nn.Sequential(
        #     InceptionBlock(self.encoder_channel + 3, 256, 64, 128, 32, 64, 128, 3)
        #     # InceptionBlock(self.encoder_channel + 3, 512, 128, 256, 64, 128, 128, 512),
        #     # InceptionBlock(512, 128, 64, 128, 64, 32, 32, 3)

        # )

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 N
        # self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # 1 2 N

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        bs , n , _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        # folding decoder
        fd1, fd2 = self.decoder(feature_global) # B N 3
        return (fd2, fd2) # FoldingNet producing final result directly
        
    def decoder(self,x):
        num_sample = self.grid_size * self.grid_size
        bs = x.size(0)
        features = x.view(bs, self.encoder_channel, 1).expand(bs, self.encoder_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd1.transpose(2,1).contiguous() , fd2.transpose(2,1).contiguous()
        # return fd2.transpose(2,1).contiguous()