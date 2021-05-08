from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model.functions import *

class generator(nn.Module):
    def __init__(self,gene_cfg,input_feature=3):
        super(generator,self).__init__()
        self.Network=nn.Sequential()
        self.input_feature=input_feature
        for i,layers in enumerate(gene_cfg):
            if layers['type']=='CNN':

                self.Network.add_module(
                    name='CNN'+str(i),
                    module=nn.Conv2d(
                        in_channels=self.input_feature,
                        out_channels=int(layers['filters']),
                        kernel_size=int(layers['size']),
                        stride=int(layers['stride']),
                        padding=(int(layers['size'])-1)//2,
                        bias=False
                    )
                )

                self.input_feature=int(layers['filters'])

            elif layers['type']=='Block':

                self.Network.add_module(
                    name='Block'+str(i),
                    module=single_block(
                        in_channels=self.input_feature,
                        filters=int(layers['filters']),
                        size=int(layers['size']),
                        stride=int(layers['stride'])
                    )
                )

                self.input_feature=int(layers['filters'])

            elif layers['type']=='Pixel':

                self.Network.add_module(
                    name='Deconv'+str(i),
                    module=nn.PixelShuffle(
                        upscale_factor=int(layers['scale'])
                    )
                )

                self.input_feature=self.input_feature//int(layers['scale'])**2

            elif layers['type']=='Activate':

                self.Network.add_module(
                    name='Activate'+str(i),
                    module=nn.ReLU()
                )

        self.end=nn.Sequential()

        self.end.add_module(
                name='CNN',
                module=nn.Conv2d(
                    in_channels=self.input_feature,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
            )
        )

        self.add_module(
            name='BN',
            module=nn.BatchNorm2d(3)
        )

        self.end.add_module(
            name='tanh',
            module=nn.Tanh()
        )

    def forward(self,x,targets=None):
        x=self.Network(x)
        x=self.end(x)
        return x

class my_linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(my_linear,self).__init__()

        self.regression=nn.Linear(
            in_features=in_features,
            out_features=out_features
        )

    def forward(self,x):
        x=x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        x=self.regression(x)
        return x


class decode(nn.Module):
    def __init__(self,decode_cfg,image_size):
        super(decode,self).__init__()
        self.decode=[]
        self.Network=nn.Sequential()
        self.previous_layer_feature=3
        for i,layers in enumerate(decode_cfg):
            if layers['type']=='CNN':

                self.Network.add_module(
                    name='CNN'+str(i),
                    module=nn.Conv2d(
                        in_channels=self.previous_layer_feature,
                        out_channels=int(layers['filters']),
                        kernel_size=int(layers['size']),
                        stride=int(layers['stride']),
                        padding=(int(layers['size'])-1)//2,
                    )
                )

                self.previous_layer_feature=int(layers['filters'])

                if int(layers['stride'])>1:
                    self.decode.append(self.Network)
                    self.Network=nn.Sequential()

            elif layers['type']=='Activate':
                self.Network.add_module(
                    name='Avtivate'+str(i),
                    module=nn.LeakyReLU(0.01)
                )

            elif layers['type']=='Tanh':
                self.Network.add_module(
                    name='Tanh'+str(i),
                    module=nn.Tanh()
                )

            elif layers['type']=='BN':
                self.Network.add_module(
                    name='BN'+str(i),
                    module=nn.BatchNorm2d(
                        num_features=self.previous_layer_feature
                    )
                )
        self.decode1=self.decode[0]
        self.decode2=self.decode[1]
        self.decode3=self.decode[2]
        self.decode4=self.decode[3]
        self.decode5=self.decode[4]
        self.decode6=self.Network

        self.Network=nn.Sequential()

        self.Network.add_module(
            name='linear',
            module=nn.Linear(
                in_features=512*8*8,
                out_features=256
            )
        )

        self.Network.add_module(
            name='actiavte',
            module=nn.LeakyReLU(0.01)
        )

    def forward(self,x,targets=None):
        input_x=x

        x1=self.decode1(x)

        x2=self.decode2(x1)

        x3=self.decode3(x2)

        x4=self.decode4(x3)

        x5=self.decode5(x4)

        x6=self.decode6(x5)

        x=x6.view(x6.shape[0],x6.shape[1]*x6.shape[2]*x6.shape[3])

        x=self.Network(x)
        if targets is not None:
            loss_function=nn.MSELoss()
            loss=loss_function(input_x,targets)

            with torch.no_grad():
                targets=self.decode1(targets)
            loss+=loss_function(x1,targets)

            with torch.no_grad():
                targets=self.decode2(targets)
            loss+=loss_function(x2,targets)

            with torch.no_grad():
                targets=self.decode3(targets)
            loss+=loss_function(x3,targets)

            with torch.no_grad():
                targets=self.decode4(targets)
            loss+=loss_function(x4,targets)

            with torch.no_grad():
                targets=self.decode5(targets)
            loss+=loss_function(x5,targets)

            with torch.no_grad():
                targets=self.decode6(targets)
            loss+=loss_function(x6,targets)
            targets=targets.view(targets.shape[0],targets.shape[1]*targets.shape[2]*targets.shape[3])

            with torch.no_grad():
                targets=self.Network(targets)
            loss+=loss_function(x,targets)
        return (x,loss) if targets is not None else x


class single_block(nn.Module):
    def __init__(self,in_channels,filters,size,stride):
        super(single_block,self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=size,
            stride=stride,
            padding=(size-1)//2,
            bias=False
        )

        self.bn1=nn.BatchNorm2d(
            num_features=filters
        )

        self.activate=nn.LeakyReLU(0.01)

        self.conv2=nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=size,
            stride=stride,
            padding=(size-1)//2,
            bias=False
        )

        self.bn2=nn.BatchNorm2d(
            num_features=filters
        )

    def forward(self,x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=self.activate(y)
        y=self.conv2(y)
        y=self.bn2(y)
        x=torch.add(x,y)
        return x

class decision(nn.Module):
    def __init__(self,decision_cfg):
        super(decision,self).__init__()
        self.Network=nn.Sequential()
        self.input_feature=3
        for i,layers in enumerate(decision_cfg):
            if layers['type']=='CNN':

                self.Network.add_module(
                    name='CNN'+str(i),
                    module=nn.Conv2d(
                        in_channels=self.input_feature,
                        out_channels=int(layers['filters']),
                        kernel_size=int(layers['size']),
                        stride=int(layers['stride']),
                        padding=(int(layers['size'])-1)//2,
                        bias=False
                    )
                )

                self.input_feature=int(layers['filters'])

            elif layers['type']=='Block':

                self.Network.add_module(
                    name='Block'+str(i),
                    module=single_block(
                        in_channels=self.input_feature,
                        filters=int(layers['filters']),
                        size=int(layers['size']),
                        stride=int(layers['stride'])
                    )
                )

                self.input_feature=int(layers['filters'])

            elif layers['type']=='Pixel':

                self.Network.add_module(
                    name='Pixel'+str(i),
                    module=nn.PixelShuffle(
                        upscale_factor=int(layers['scale'])
                    )
                )

                self.input_feature=self.input_feature//int(layers['scale'])**2

            elif layers['type']=='Activate':

                self.Network.add_module(
                    name='Activate'+str(i),
                    module=nn.LeakyReLU(0.01)
                )

        self.end=nn.Sequential()

        self.end.add_module(
                name='CNN',
                module=nn.Conv2d(
                    in_channels=self.input_feature,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
            )
        )

        self.add_module(
            name='BN',
            module=nn.BatchNorm2d(3)
        )

        self.end.add_module(
            name='tanh',
            module=nn.Tanh()
        )

    def forward(self,x,targets=None):
        x=self.Network(x)
        x=self.end(x)
        loss=nn.MSELoss()
        if targets is not None:
            loss=loss(x,targets)
        return (x,loss) if targets is not None else x
