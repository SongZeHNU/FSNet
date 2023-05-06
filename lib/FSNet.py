import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_encoder import vitencoder 
from config import get_config
import math
from torch.nn.parameter import Parameter
import scipy.stats as st
import numpy as np
from torch.nn import BatchNorm2d as bn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual


class EEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EEM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 3, padding = 1),
        )
    def forward(self, x):

        x = self.relu(self.branch0(x))
        return x


class CrossConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(CrossConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.downsample = F.interpolate(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_upsample41 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample42 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample43 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_downsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1 * self.conv_downsample1(F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=True)) 
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 * self.conv_downsample2(F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=True))
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3  
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x, x1_1, x2_2, x3_2


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', bn(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', bn(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class Network(nn.Module):
    def __init__(self, args, channel=32 ):
        super(Network, self).__init__()
        
        config = get_config(args)
        self.swin = vitencoder(config, img_size=args.trainsize)
        self.swin.load_from(config)

        #conv3x3 to channel 32
        self.eem1_1 = EEM(128, channel)
        self.eem2_1 = EEM(256, channel)
        self.eem3_1 = EEM(512, channel)
        self.eem4_1 = EEM(2048, channel)
        # cross Decoder 
        self.CCD = CrossConnectionDecoder(channel)

        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(12*channel, 1, 1)                                         
        self.cath1 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)                     
        self.conv_concat3 = BasicConv2d(5*channel, 5*channel, 3, padding=1)                   
        self.conv6 = BasicConv2d(5*channel, 5*channel, 3, padding=1)                          
        self.conv7 = nn.Conv2d(20*channel, 1, 1)                                               
    

        self.conv_upsample6 = BasicConv2d(5*channel, 5*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(9*channel, 9*channel, 3, padding=1)
        self.conv8 = BasicConv2d(9*channel, 9*channel, 3, padding=1)
        self.conv9 = nn.Conv2d(39*channel, 1, 1)                                             
    

        self.conv_upsample7 = BasicConv2d(9*channel, 9*channel, 3, padding=1)
        self.conv_concat5 = BasicConv2d(10*channel, 10*channel, 3, padding=1)
        self.conv10 = BasicConv2d(10*channel, 10*channel, 3, padding=1)
        self.conv11 = nn.Conv2d(85*channel, 1, 1)                                          
    


        self.refunet = RefUnet(1,64)
        self.refunet2 = RefUnet(1,64)
    

        dropout0 = 0.1
        d_feature0 = 128
        d_feature1 =64
        num = 64
        self.ASPP_3_1 = _DenseAsppBlock(input_num=num, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)
        self.ASPP_6_1 = _DenseAsppBlock(input_num=num + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)
        self.ASPP_12_1 = _DenseAsppBlock(input_num=num + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)
        self.ASPP_18_1 = _DenseAsppBlock(input_num=num + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)
        self.ASPP_24_1 = _DenseAsppBlock(input_num=num + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        
        d_feature0 = 256
        d_feature1 =96
        num = 160
        self.ASPP_3_2 = _DenseAsppBlock(input_num=num, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)
        self.ASPP_6_2 = _DenseAsppBlock(input_num=num + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)
        self.ASPP_12_2 = _DenseAsppBlock(input_num=num + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)
        self.ASPP_18_2 = _DenseAsppBlock(input_num=num + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)
        self.ASPP_24_2 = _DenseAsppBlock(input_num=num + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        d_feature0 = 512
        d_feature1 =192
        num = 288
        self.ASPP_3_3 = _DenseAsppBlock(input_num=num, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)
        self.ASPP_6_3 = _DenseAsppBlock(input_num=num + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)
        self.ASPP_12_3 = _DenseAsppBlock(input_num=num + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)
        self.ASPP_18_3 = _DenseAsppBlock(input_num=num + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)
        self.ASPP_24_3 = _DenseAsppBlock(input_num=num + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        d_feature0 = 1024
        d_feature1 =480
        num = 320
        self.ASPP_3_4 = _DenseAsppBlock(input_num=num, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)
        self.ASPP_6_4 = _DenseAsppBlock(input_num=num + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)
        self.ASPP_12_4 = _DenseAsppBlock(input_num=num + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)
        self.ASPP_18_4 = _DenseAsppBlock(input_num=num + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)
        self.ASPP_24_4 = _DenseAsppBlock(input_num=num + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        d_feature0 = 1280
        d_feature1 =512
        num = 320
        self.ASPP_3_5 = _DenseAsppBlock(input_num=num, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)
        self.ASPP_6_5 = _DenseAsppBlock(input_num=num + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)
        self.ASPP_12_5 = _DenseAsppBlock(input_num=num + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)
        self.ASPP_18_5 = _DenseAsppBlock(input_num=num + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)
        self.ASPP_24_5 = _DenseAsppBlock(input_num=num + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
   
    def forward(self, x):
        # Feature Extraction
        x = self.swin(x)

        x1 = x[4]                        # bs, 128, 96, 96
        _, pix, channel = x1.size()
        x1 = x1.transpose(1,2)
        x1 = x1.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x2 = x[0]                        # bs, 256, 48, 48    
        _, pix, channel = x2.size()
        x2 = x2.transpose(1,2)
        x2 = x2.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x3 = x[1]                        # bs, 512, 24, 24
        _, pix, channel = x3.size()
        x3 = x3.transpose(1,2)
        x3 = x3.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

        x4 = torch.cat([x[2],x[3]],dim = 2)                        # bs, 1024, 12, 12
        _, pix, channel = x4.size()
        x4 = x4.transpose(1,2)
        x4 = x4.view(-1, channel, int(math.sqrt(pix)), int(math.sqrt(pix))) 

       

        b1 = self.eem1_1(x1)        # channel -> 32
        b2 = self.eem2_1(x2)        # channel -> 32
        b3 = self.eem3_1(x3)        # channel -> 32
        b4 = self.eem4_1(x4)        # channel -> 32
     

        S_g, h3, h2 ,h1 = self.CCD(b4, b3, b2)
        #coarse map
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    

        sg1 = S_g_pred
        S_g_pred = self.refunet2(S_g_pred)
        
     
        x4_1 = self.conv4(b4)

        x4_1 = torch.cat([x4_1,h3], dim=1)
        x4_1 = self.cath1(x4_1)

        aspp3 = self.ASPP_3_1(x4_1)
        feature = torch.cat((aspp3, x4_1), dim=1)

        aspp6 = self.ASPP_6_1(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12_1(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18_1(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24_1(feature)
        feature = torch.cat((aspp24, feature), dim=1)
        ra4_feat = self.conv5(feature)

        S_5 = ra4_feat 
        #m4
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  

        x3_2 = torch.cat((b3, self.conv_upsample5(self.upsample(x4_1)),h2), 1)
        x3_1 = self.conv_concat3(x3_2)

        x3_1 = self.conv6(x3_1)
        aspp3 = self.ASPP_3_2(x3_1)
        feature = torch.cat((aspp3, x3_1), dim=1)

        aspp6 = self.ASPP_6_2(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12_2(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18_2(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24_2(feature)
        feature = torch.cat((aspp24, feature), dim=1)
        ra3_feat = self.conv7(feature)


        S_4 = ra3_feat 
        #m3
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear') 
    
        x2_2 = torch.cat((b2, self.conv_upsample6(self.upsample(x3_1)),h1), 1)
        x2_1 = self.conv_concat4(x2_2)
        x2_1 = self.conv8(x2_1)
        aspp3 = self.ASPP_3_3(x2_1)
        feature = torch.cat((aspp3, x2_1), dim=1)

        aspp6 = self.ASPP_6_3(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12_3(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18_3(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24_3(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        ra2_feat = self.conv9(feature)

        S_3 = ra2_feat
        #m2
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   


        x1_2 = torch.cat((b1, self.conv_upsample7(self.upsample(x2_1))), 1)
        x1_1 = self.conv_concat5(x1_2)
        x1_1 = self.conv10(x1_1)
        aspp3 = self.ASPP_3_4(x1_1)
        feature = torch.cat((aspp3, x1_1), dim=1)

        aspp6 = self.ASPP_6_4(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12_4(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18_4(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24_4(feature)
        feature = torch.cat((aspp24, feature), dim=1)
        ra1_feat = self.conv11(feature)
     
        S_2 = ra1_feat 
        #m1
        S_2_pred = F.interpolate(S_2, scale_factor=4, mode='bilinear')   
        sg2 = S_2_pred 
        S_2_pred = self.refunet(S_2_pred) 

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred, sg1, sg2 , S_2_pred 

