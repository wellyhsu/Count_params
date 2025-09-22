# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file densestack.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief DenseStack
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append('/media/Pluto/Hao/HandMesh_origin/mobrecon')

import torchvision.models as models
import torch
import torch.nn as nn
from mobrecon.models.modules import conv_layer, mobile_unit, linear_layer, Reorg
import os
import numpy as np
from thop import clever_format, profile
from torchinfo import summary

# from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import resnet18, ResNet18_Weights
from mobrecon.models.optimized_mobileV3 import MobileNetV3_optimized, HSwish

class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in//4)
        self.conv2 = mobile_unit(channel_in*5//4, channel_in//4)
        self.conv3 = mobile_unit(channel_in*6//4, channel_in//4)
        self.conv4 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3),dim=1)
        out4 = self.conv4(comb3)
        comb4 = torch.cat((comb3, out4),dim=1)
        return comb4


class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in//2)
        self.conv2 = mobile_unit(channel_in*3//2, channel_in//2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        return comb2


class DenseBlock3(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock3, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in)
        self.conv2 = mobile_unit(channel_in*2, channel_in)
        self.conv3 = mobile_unit(channel_in*3, channel_in)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3),dim=1)
        return comb3


class DenseBlock2_noExpand(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_noExpand, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in*3//4)
        self.conv2 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((out1, out2),dim=1)
        return comb2


class SenetBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel, size):
        super(SenetBlock, self).__init__()
        self.size = size
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel = channel
        self.fc1 = linear_layer(self.channel, min(self.channel//2, 256))
        self.fc2 = linear_layer(min(self.channel//2, 256), self.channel, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_out = x
        pool = self.globalAvgPool(x)
        pool = pool.view(pool.size(0), -1)
        fc1 = self.fc1(pool)
        out = self.fc2(fc1)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out * original_out


class DenseStack(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel):
        super(DenseStack, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel*2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel*2)
        self.senet2 = SenetBlock(input_channel*4,16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel*4)
        self.senet3 = SenetBlock(input_channel*8,8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel*8)
        self.dense5 = DenseBlock2_noExpand(input_channel*8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel*8, input_channel*4, num3x3=1), mobile_unit(input_channel*4, input_channel*4, num3x3=2))
        self.senet4 = SenetBlock(input_channel*4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink2 = nn.Sequential(mobile_unit(input_channel*4, input_channel*2, num3x3=1), mobile_unit(input_channel*2, input_channel*2, num3x3=2))
        self.senet5 = SenetBlock(input_channel*2,8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink3 = nn.Sequential(mobile_unit(input_channel*2, input_channel*2, num3x3=1), mobile_unit(input_channel*2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel,16)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.dense3(d2))
        u1 = self.upsample1(self.senet4(self.thrink1(d3)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.upsample3(self.senet6(self.thrink3(us2)))
        return u3


class DenseStack2(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel, final_upsample=True, ret_mid=False):
        super(DenseStack2, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel*2,32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel*2)
        self.senet2 = SenetBlock(input_channel*4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel*4)
        self.senet3 = SenetBlock(input_channel*8,8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel*8)
        self.dense5 = DenseBlock2_noExpand(input_channel*8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel*8, input_channel*4, num3x3=1), mobile_unit(input_channel*4, input_channel*4, num3x3=2))
        self.senet4 = SenetBlock(input_channel*4,4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink2 = nn.Sequential(mobile_unit(input_channel*4, input_channel*2, num3x3=1), mobile_unit(input_channel*2, input_channel*2, num3x3=2))
        self.senet5 = SenetBlock(input_channel*2,8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink3 = nn.Sequential(mobile_unit(input_channel*2, input_channel*2, num3x3=1), mobile_unit(input_channel*2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel,16)
        self.final_upsample = final_upsample
        if self.final_upsample:
            self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ret_mid = ret_mid

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.senet3(self.dense3(d2)))
        d4 = self.dense5(self.dense4(d3))
        u1 = self.upsample1(self.senet4(self.thrink1(d4)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.senet6(self.thrink3(us2))
        if self.final_upsample:
            u3 = self.upsample3(u3)
        if self.ret_mid:
            return u3, u2, u1, d4
        else:
            return u3, d4


class DenseStack_Backnone(nn.Module):   # load weight shape error -> pretrain=False
    def __init__(self, input_channel=128, out_channel=24, latent_size=256, kpts_num=21, pretrain=True, active_groups: int = 1):
        """Init a DenseStack

        Args:
            input_channel (int, optional): the first-layer channel size. Defaults to 128.
            out_channel (int, optional): output channel size. Defaults to 24.
            latent_size (int, optional): middle-feature channel size. Defaults to 256.
            kpts_num (int, optional): amount of 2D landmark. Defaults to 21.
            pretrain (bool, optional): use pretrain weight or not. Defaults to True.
        """
        super(DenseStack_Backnone, self).__init__()
        # # # =====================================================
        # # mobileNetV3 Optimized
        # # # =====================================================
        # self.mobilenetv3_optimized = MobileNetV3_optimized(active_groups=active_groups)
        # self.conv_branch3 = nn.Sequential(
        #     nn.Conv2d(576, 512, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     HSwish(inplace=True),
        #     nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     HSwish(inplace=True)
        # )
        # self.conv_branch4 = nn.Sequential(
        #     nn.Conv2d(576, 512, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     HSwish(inplace=True),
        #     nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     HSwish(inplace=True),
        #     nn.Conv2d(256, 21, 1, stride=1, bias=False)
        # )
        # # self.fc = nn.Linear(21 * 2 * 2, 21 * 2) # 1344x84 or 
        # self.fc = nn.Linear(4, 2)

        # # # =====================================================
        # # mobileNetV3
        # # # =====================================================
        # mm = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # self.mobilenetv3 = mm.features
        # self.conv_branch3 = nn.Sequential(
        #     nn.Conv2d(576, 512, kernel_size=3, stride=1, padding=1),  # Change 576 to 256
        #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # )
        # # Adjusting the second convolutional branch input channels
        # self.conv_branch4 = nn.Sequential(
        #     nn.Conv2d(576, 512, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(256, 21, kernel_size=1, stride=1),
        # )
        # self.fc = nn.Linear(4,2) # for mobileNet V3 small and Densestack

        # ====================================
        # Convert mobilenet_v3_small to Res18
        # Convert mobilenet_v3_small to Res18
        # Convert mobilenet_v3_small to Res18
        # ====================================
        mm = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18 = nn.Sequential(*list(mm.children())[:-2])  # 移除 ResNet 最後的全連接層
        # 調整輸入通道數量為 ResNet-18 最後一層輸出的 512
        self.conv_branch3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        self.conv_branch4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 21, kernel_size=1, stride=1),
        )
        self.fc = nn.Linear(4,2) 

    def forward(self, x):

        # # print("000000000000000000000000000000000000000print shape:")
        # # # =====================================================
        # # mobileNetV3 Optimized
        # # # =====================================================
        # mobileNetV3_output = self.mobilenetv3_optimized(x)  # Shape: (1, 192, 4, 4)
        # # print(mobileNetV3_output.shape)
        # # import pdb; pdb.set_trace()
        # latent = self.conv_branch3(mobileNetV3_output)  # Shape: (1, 256, 4, 4)
        # # print(latent.shape)
        # uv_reg = self.conv_branch4(mobileNetV3_output)  # Shape: (1, 21, 2, 2)
        # # print(uv_reg.shape)
        # uv_reg = uv_reg.contiguous().view(uv_reg.size(0), uv_reg.size(1), -1)  # Shape: (1, 21, 4)
        # # print(uv_reg.shape)
        # uv_reg = self.fc(uv_reg)  # Shape: (1, 21, 2)

        # return latent, uv_reg

        # # # =====================================================
        # # mobileNetV3
        # # # =====================================================
        # mobileNetV3_output = self.mobilenetv3(x)  # Shape: (1, 192, 4, 4)
        # latent = self.conv_branch3[0](mobileNetV3_output)
        # latent = self.conv_branch3[1](latent)
        # uv_reg = self.conv_branch4(mobileNetV3_output)
        # # uv_reg = uv_reg.view(uv_reg.shape[0], uv_reg.shape[1], -1)
        # # replace "view" as "reshape" because quantized model aren't continuous 
        # # 確保量化後的數據流不會被頻繁重新排
        # uv_reg = uv_reg.contiguous().reshape(uv_reg.shape[0], uv_reg.shape[1], -1)
        # uv_reg = self.fc(uv_reg)
        # return latent, uv_reg
    

        # ====================================
        # Convert mobilenet_v3_small to Res18
        # Convert mobilenet_v3_small to Res18
        # Convert mobilenet_v3_small to Res18
        # ====================================
        resnet18_output = self.resnet18(x).contiguous()
        latent = self.conv_branch3[0](resnet18_output)
        latent = self.conv_branch3[1](latent)
        uv_reg = self.conv_branch4(resnet18_output)
        # uv_reg = uv_reg.view(uv_reg.shape[0], uv_reg.shape[1], -1)
        # replace "view" as "reshape" because quantized model aren't continuous 
        # 確保量化後的數據流不會被頻繁重新排
        uv_reg = uv_reg.contiguous().reshape(uv_reg.shape[0], uv_reg.shape[1], -1)
        uv_reg = self.fc(uv_reg)
        # print(f"uv_reg: {uv_reg.shape}")
        # print(f"latent: {latent.shape}")
        return latent, uv_reg
    



class EdgeFriendlyBackbone(nn.Module):
    def __init__(self, kpts_num=21):
        super(EdgeFriendlyBackbone, self).__init__()

        def conv_bn_relu(in_c, out_c, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        def conv1x1_bn_relu(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        

        # 通道數設定：依照規則，Ch_out=512 => Ch_in <= 48，Ch_out <= 256 => Ch_in <= 112
        self.stage1 = nn.Sequential(
            conv_bn_relu(3, 48),  # in: 128x128x3 -> 128x128x48
            conv_bn_relu(48, 48),
            nn.MaxPool2d(2)       # -> 64x64x48
        )

        self.stage2 = nn.Sequential(
            conv_bn_relu(48, 112),
            conv_bn_relu(112, 112),
            nn.MaxPool2d(2)       # -> 32x32x112
        )

        self.stage3 = nn.Sequential(
            conv_bn_relu(112, 256),
            conv1x1_bn_relu(256, 112),
            conv_bn_relu(112, 256),
            nn.MaxPool2d(2)       # -> 16x16x256
        )

        self.stage4 = nn.Sequential(
            conv1x1_bn_relu(256, 48),
            conv_bn_relu(48, 512),
            conv1x1_bn_relu(512, 48),
            conv_bn_relu(48, 512),
            nn.MaxPool2d(2)   # -> 8×8×512
        )

        self.stage5 = nn.Sequential(
            conv1x1_bn_relu(512, 48),
            conv_bn_relu(48, 512),
            conv1x1_bn_relu(512, 48),
            conv_bn_relu(48, 512),
            nn.MaxPool2d(2)             # -> 4×4×512
        )

        self.conv_branch3 = nn.Sequential(
            conv1x1_bn_relu(512, 112),
            conv_bn_relu(112, 256),
        )  # -> 4×4×256

        self.head_proj = nn.Conv2d(256, kpts_num, kernel_size=1)

        self.fc = nn.Linear(16, 2)  # for each keypoint: 4 values to 2D regression


    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)   # 4x4x512
        latent = self.conv_branch3(x) # (B,256,4,4)
        uv_reg = self.head_proj(latent) # (B,kpts,4,4)
        uv_reg = uv_reg.contiguous().reshape(uv_reg.shape[0], uv_reg.shape[1], -1)# (B,kpts,16)
        uv_reg = self.fc(uv_reg) # (B,kpts,2)
        return latent, uv_reg
