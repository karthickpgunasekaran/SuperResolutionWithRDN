import torch
import torch.nn as nn
#import torch.nn.Functional as F

class DenseBlock(nn.Module):
    def __init__(self,in_channel_size):
        super(DenseBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel_size,out_channels=32,kernel_size=3,padding=1,bias=True,padding_mode='zeros')
        self.relu = nn.ReLU()
    def forward(self,inp):
        out = self.conv(inp)
        rel_out = self.relu(out)
        res = torch.cat((inp,out),1)
        return res

class ResidualDenseBlock(nn.Module):
    def __init__(self):
        super(ResidualDenseBlock, self).__init__()
        self.dense_block = nn.Sequential(DenseBlock(64),DenseBlock(96),DenseBlock(128),DenseBlock(160),DenseBlock(192),DenseBlock(224))
        self.conv = nn.Conv2d(in_channels=256,out_channels=64,kernel_size=1,bias=True)
    def forward(self,inp):
        dense_out = self.dense_block(inp)
        conv_out = self.conv(dense_out)
        out = conv_out+inp
        return out

class ResidualDenseNetwork(nn.Module):
    def __init__(self,scale=1):
        super(ResidualDenseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,bias=True,padding=1,padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=True, padding=1,padding_mode='zeros')
        #Init residual dense block
        self.residual_block = ResidualDenseBlock()
        #Global Feature Fusion
        self.feature_fusion_1 = nn.Conv2d(in_channels=64*3,out_channels=64,kernel_size=1,bias=True)
        self.feature_fusion_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,padding_mode='zeros',bias=True)
        #Upsampling
        self.conv3 =  nn.Conv2d(in_channels=64,out_channels=64*scale*scale,kernel_size=3,bias=True,padding=1,padding_mode='zeros')
        self.up_scale = nn.PixelShuffle(scale)
        #Final convolution
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1,padding_mode='zeros',bias=True)
    def forward(self,inp):
        #print("inp:",inp.shape)
        F_neg1 = self.conv1(inp)
        #print("F_neg1 : ",F_neg1.shape)
        F_0 = self.conv2(F_neg1)
        F_1 = self.residual_block(F_0)
        F_2 =self.residual_block(F_1)
        F_3 = self.residual_block(F_2)
        F_D = torch.cat((F_1,F_2,F_3),1)
        F_conv1 = self.feature_fusion_1(F_D)
        F_GF = self.feature_fusion_2(F_conv1)
        F_DF = F_GF+F_neg1
        F_conv2 = self.conv3(F_DF)
        F_up = self.up_scale(F_conv2)
        F_hr = self.conv4(F_up)
        return F_hr
