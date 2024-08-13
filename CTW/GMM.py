"""
GMM.py is modified from https://github.com/minar09/cp-vton-plus
"""

from torch.nn import init
from torch import nn
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=2, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(
                in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(out_ngf, out_ngf, kernel_size=3,
                            stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(out_ngf, out_ngf, kernel_size=3,
                            stride=1, padding=1), nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                         epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h*w)
        feature_B = feature_B.view(b, c, h*w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(
            b, h, w, h*w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=256, output_dim=512, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 8 * 6, 512 * 3)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        x = x.view(-1, 3, 512)
        return x

class FeatureConv(nn.Module):
    def __init__(self, input_nc=768, output_nc=256, use_cuda=True):
        super(FeatureConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        )
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        x = self.conv(x)
        return x

class GMM(nn.Module):
    """ Geometric Matching Module
    """

    def __init__(self, channelA=4, channelB=3):
        super(GMM, self).__init__()
        n_layers = 2
        ngf = 64
        self.extractionA = FeatureExtraction(
            channelA, ngf=ngf, n_layers=n_layers, norm_layer=nn.BatchNorm2d)
        self.extractionB = FeatureExtraction(
            channelB, ngf=ngf, n_layers=n_layers, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(
            input_nc=768, output_dim=512, use_cuda=True)

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)

        return self.regression(correlation)


class FeatureRegression2(nn.Module):
    def __init__(self, input_nc=256, output_dim=512, use_cuda=True):
        super(FeatureRegression2, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_nc, input_nc, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(input_nc),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(input_nc, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
        self.linear = nn.Linear(512 * 4 * 3, 512 * 3)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
#         x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        x = x.view(-1, 3, 512)
        return x


class Encoder(nn.Module):
    """ Encoder
    """

    def __init__(self, channelA=4):
        super(Encoder, self).__init__()
        n_layers = 5
        ngf = 64
        self.extractionA = FeatureExtraction(
            channelA, ngf=ngf, n_layers=n_layers, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.regression = FeatureRegression2(
            input_nc=512, output_dim=512, use_cuda=True)

    def forward(self, inputA):
        featureA = self.extractionA(inputA)
        featureA = self.l2norm(featureA)

        return self.regression(featureA)
