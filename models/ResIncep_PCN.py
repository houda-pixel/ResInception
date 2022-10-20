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
class ResIncep_PCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_fine = config.num_pred
        self.encoder_channel = config.encoder_channel
        self.encoder_channel_3 = 3072
        grid_size = 4 # set default

        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2 )
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
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )
        # self.final_conv = nn.Sequential(
        #     nn.Conv1d(1024+3+2,512,1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512,512,1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512,3,1)
        # )

        self.final_conv = nn.Sequential(
            InceptionBlock(1024+3+2, 512, 128, 256, 64, 128, 128, 512),
            InceptionBlock(512, 128, 64, 128, 64, 32, 32, 3)

        )


        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda() # 1 2 S
        # self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2) # 1 2 S
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):

        # print(xyz.shape)
        bs , n , _ = xyz.shape
        # encoder
        #xyz = torch.flip(xyz, dims=(1,))
        xyz = xyz[:, torch.randperm(xyz.shape[1]), :]
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        
        # print("feature_global", feature_global.shape)


        # decoder
        coarse = self.mlp(feature_global).reshape(-1,self.number_coarse,3) # B M 3
        # print("coarse", coarse.shape)

        point_feat = coarse.unsqueeze(2).expand(-1,-1,self.grid_size**2,-1) # B M S 3
        point_feat = point_feat.reshape(-1,self.number_fine,3).transpose(2,1) # B 3 N
        # print("point_feat", point_feat.shape)


        seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S
        seed = seed.reshape(bs,-1,self.number_fine)  # B 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1,-1,self.number_fine) # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # B C N
        
        # print("feat", feat.shape)
        fine = self.final_conv(feat) + point_feat   # B 3 N

        # print("fine", fine.shape)


        return (coarse.contiguous(), fine.transpose(1,2).contiguous())
