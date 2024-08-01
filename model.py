from typing import Dict

import torch
from torch import nn
from collections import OrderedDict

class My_Model(nn.Module):
    def __init__(self, hparams:Dict={}, only_output_target=False):
        super(My_Model, self).__init__()
        hparams.setdefault('feature_extract_channels', 2)
        hparams.setdefault('out_channels', 1)
        hparams.setdefault('in_channels', 3)
        hparams.setdefault('layers', 2)

        self.feature_model = Feature_Model(
            channels=hparams['feature_extract_channels'],
            out_channels=hparams['out_channels'],
            in_channels=hparams['in_channels'],
            layers=hparams['layers']
            )

        self.k_max = 256
        self.out_channels = hparams['out_channels']
        self.crm = Sigmoid_CRM(a=0.6, b=0.9)
        self.forward = self.forward_only_tar if only_output_target else self.forward

    def forward(self, p0):
        e0 = self.crm.crf_inv(p0)
        K_temp = self.feature_model(e0)
        K = torch.clamp(K_temp, max=self.k_max, min=0.000001)
        if self.out_channels == 1:
            K = K.repeat(1, 3, 1, 1)
        p1 = self.crm.btf(p0, K)
        p1 = torch.clamp(p1, max=1)
        return p1, K

    def forward_only_tar(self, p0):
        e0 = self.crm.crf_inv(p0)
        K_temp = self.feature_model(e0)
        K = torch.clamp(K_temp, max=self.k_max, min=0.000001)
        if self.out_channels == 1:
            K = K.repeat(1, 3, 1, 1)
        p1 = self.crm.btf(p0, K)
        p1 = torch.clamp(p1, max=1)
        return p1



# Common CRM
class CRM():
    def __init__(self):
        # 建立模型所需的参数
        pass

    def init_params(self):
        # 初始化参数状态
        pass

    def crf(self, E):
        # 实际响应到相机输出图像之间的变换
        pass

    def btf(self, E, K):
        # 相机输出的低照度图像到预期的正常照度图像之间的变换
        pass

    def crf_inv(self, E):
        # 相机输出图像到实际响应之间的变换
        pass

# sigmoid_CRM
class Sigmoid_CRM(CRM):
    def __init__(self, a=0.6, b=0.9):
        super(Sigmoid_CRM, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1, requires_grad=False))
        self.b = torch.nn.Parameter(torch.randn(1, requires_grad=False))
        self.init_params(a, b)

    def init_params(self, a, b):
        self.a = a
        self.b = b

    def crf(self, E):
        return (1 + self.a)*(torch.pow(E, self.b)/(torch.pow(E, self.b) + self.a))

    def btf(self, E, K):
        return (torch.pow(K, self.b)*E*(1+self.a))/((torch.pow(K, self.b)-1)*E+1+self.a)

    def crf_inv(self, E):
        return torch.pow((self.a*E)/(1+self.a-E), 1/self.b)


class FeatureBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FeatureBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            padding=1,
            bias=1
        )
        self.convblock = self.conv

    def forward(self, input):
        out = self.convblock(input)
        return out

class Feature_Model(nn.Module):
    def __init__(self, channels=2, out_channels=3, in_channels=3, layers=2):
        super(Feature_Model, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=1e-2)
        self.layers = OrderedDict()
        if layers == 1:
            self.layers['1'] = FeatureBlock(in_channels, out_channels)
        if layers >= 2:
            self.layers['1'] = FeatureBlock(in_channels, channels)
            self.layers['2'] = self.relu
            for i in range(layers-2):
                self.layers[str(3+i*2-1)]=FeatureBlock(channels, channels)
                self.layers[str(3+i*2)]=self.relu
            self.layers['-1'] = FeatureBlock(channels, out_channels)
        self.model = nn.Sequential(self.layers)

    def forward(self, x):
        x=self.model(x)
        return x


# for self test
if __name__ == '__main__':
    input = torch.randn((1, 3, 100, 100))
    device = torch.device('cpu')
    hparams = {'out_channels': '1'}
    a = My_Model(hparams)
    a.to(device)
    input = input.to(device)
    a(input)
    exit(0)




