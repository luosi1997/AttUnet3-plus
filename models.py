from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Vgg16_AttU_Net3P(nn.Module):
    def __init__(self, n_channels=3, num_classes=1, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features  # 加载的是fc前面的网络结构
        self.encoder[0] = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        ## -------------Encoder--------------
        self.conv1 = nn.Sequential(self.encoder[0],  # (1,64)->(n_channels,64)
                                   self.relu,
                                   self.encoder[2],  # (64,64)
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],  # (64,128)
                                   self.relu,
                                   self.encoder[7],  # (128,128)
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],  # (128,256)
                                   self.relu,
                                   self.encoder[12],  # (256,256)
                                   self.relu,
                                   self.encoder[14],  # (256,256)
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],  # (256,512)
                                   self.relu,
                                   self.encoder[19],  # (512,512)
                                   self.relu,
                                   self.encoder[21],  # (512,512)
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],            # (512,1024)
                                   self.relu,
                                   self.encoder[26],            # (1024,1024)
                                   self.relu,
                                   self.encoder[28],            # (1024,1024)
                                   self.relu)
        ## -------------Decoder--------------
        filters = [64, 128, 256, 512, 512]

        self.CatChannels = filters[0]  # 每个decoder用5个尺度的特征图进行拼接，每个尺度的特征图的特征通道都为64，即filter[0]
        self.CatBlocks = 5  # 每个decoder有来自五个尺度的特征图进行拼接
        self.UpChannels = self.CatChannels * self.CatBlocks  # 64*5->320 每个decoder拼接后的特征通道数量 320

        '''stage 4d'''
        # Deccoder4中，获取较小四层的详细信息的拼接操作
        # 对En1的操作 maxpooling(8), 64, 3*3
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        # MaxPool2d(kernel_size, stride, ceil_mode) kernel_size指最大池的窗口大小，stride是一次移动的步长，ceil_mode是向上取整。
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)  # padding对卷积后的特征图进行了边缘像素的修补。
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)  # 参数为特征通道的数量
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)  # inplace=True 函数会把输出直接覆盖到输入中

        # 对En2的操作，maxpooling(4), 64, 3*3
        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # 对En3的操作，maxpooling(2), 64, 3*3
        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # 对同层En4的操作，64， 3*3。 同层没有最大池的操作。
        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # Decoder4中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(2)
        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=True)  # 14*14  #scale_factor指定输出为输入的多少倍
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        # 特征聚合机制
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # 对En1的操作 maxpooling(4), 64, 3*3
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # 对En2的操作 maxpooling(2), 64, 3*3
        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # 对同层En3的操作，64， 3*3。 同层没有最大池的操作。
        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # Decoder3中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(2)
        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # Decoder3中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(4)
        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        # 特征聚合机制
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # 对En1的操作 maxpooling(2), 64, 3*3
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # 对同层En2的操作，64， 3*3。 同层没有最大池的操作。
        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # Decoder2中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(2)
        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # Decoder2中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(4)
        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # Decoder2中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(4)
        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        # 特征聚合机制
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # 对同层En1的操作，64， 3*3。 同层没有最大池的操作。
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # Decoder1中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(2)
        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # Decoder1中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(4)
        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # Decoder1中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(8)
        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # Decoder1中，获取较大层的粗粒度信息的拼接操作
        # bilinear upsample(16)
        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        # 特征聚合机制
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, num_classes, 3, padding=1)  # 最后通过3*3卷积，将320通道转为n_classes通道。

        ########################################################################################################
         # -------------Bilinear Upsampling--------------
        self.upscore32 = nn.Upsample(scale_factor=32,mode='bilinear', align_corners=True)
        self.upscore16 = nn.Upsample(scale_factor=16,mode='bilinear', align_corners=True)
        self.upscore8 = nn.Upsample(scale_factor=8,mode='bilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=4,mode='bilinear', align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Att128_64 = Attention_block(F_g=128, F_l=64, F_int=64)
        self.Att256_128 = Attention_block(F_g=256, F_l=128, F_int=128)
        self.Att512_256 = Attention_block(F_g=512, F_l=256, F_int=256)
        self.Att512_512 = Attention_block(F_g=512, F_l=512, F_int=512)
        self.Att320_256 = Attention_block(F_g=320, F_l=256, F_int=256)
        self.Att320_128 = Attention_block(F_g=320, F_l=128, F_int=128)
        self.Att320_64 = Attention_block(F_g=320, F_l=64, F_int=64)

    def forward(self, x):
        ## -------------Encoder-------------
        h1 = self.conv1(x)  # (1,320x320)->(64,320x320)
        h2 = self.conv2(self.pool(h1))  # (64,320x320)->(128,160x160)
        h3 = self.conv3(self.pool(h2))  # (128,160x160)->(256,80x80)
        h4 = self.conv4(self.pool(h3))  # (256,80x80)->(512,40x40)
        hd5 = self.conv5(self.pool(h4))  # (512,40x40)->(512,20x20)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(
            self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(self.Att128_64(g=self.upscore2(h2), x=h1)))))  # 64->64
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(
            self.h2_PT_hd4_conv(self.h2_PT_hd4(self.Att256_128(g=self.upscore2(h3), x=h2)))))  # 128->64
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(
            self.h3_PT_hd4_conv(self.h3_PT_hd4(self.Att512_256(g=self.upscore2(h4), x=h3)))))  # 256->64
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(
            self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(self.Att512_512(g=self.upscore2(hd5), x=h4))))  # 512->64
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))  # 1024->64
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(  # 64*5=320->320
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(
            self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(self.Att128_64(g=self.upscore2(h2), x=h1)))))  # 64->64
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(
            self.h2_PT_hd3_conv(self.h2_PT_hd3(self.Att256_128(g=self.upscore2(h3), x=h2)))))  # 128->64
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(
            self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(self.Att320_256(g=self.upscore2(hd4), x=h3))))  # 256->64
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))  # 512->64
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))  # 1024->64
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(  # 64*5=320->320
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(
            self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(self.Att128_64(g=self.upscore2(h2), x=h1)))))  # 64->64
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(
            self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(self.Att320_128(g=self.upscore2(hd3), x=h2))))  # 128->64
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))  # 256->64
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))  # 512->64
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))  # 1024->64
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(  # 64*5=320->320
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(
            self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(self.Att320_64(g=self.upscore2(hd2), x=h1))))  # 64->64
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))  # 128->64
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))  # 256->64
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))  # 512->64
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))  # 1024->64
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(  # 64*5=320->320
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes  320->n_classes
        return d1