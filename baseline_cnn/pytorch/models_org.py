import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchlibrosa as tl
import numpy as np
import matplotlib.pyplot as plt
import config

def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    elif 'bool' in str(x.dtype):
        x = torch.BoolTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='max', activation='relu'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if activation == 'relu':
            x = F.relu_(self.bn2(self.conv2(x)))
        elif activation == 'sigmoid':
            x = torch.sigmoid(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=stride,
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1), stride=stride,
                               padding=(0, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, pool_size=(2, 2), pool_type='max', activation='relu'):
        identity = input
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        identity = self.bn3(self.conv3(identity))

        x += identity

        if activation == 'relu':
            x = F.relu_(x)
        elif activation == 'sigmoid':
            x = torch.sigmoid(x)
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num， isres):
        super(DecisionLevelMaxPooling, self).__init__()
        sample_rate=config.sample_rate
        window_size = config.win_length
        hop_size = config.hop_length
        mel_bins = config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = tl.Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        self.logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=20, fmax=2000, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        if isres:
            self.cnn_encoder = Res_encoder()
        else:
            self.cnn_encoder = CNN_encoder()

        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input, ifplot, audio_name):
        """input: (samples_num, date_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        batch_size, channel_num, _, mel_bins = x.shape
        x_diff1 = torch.diff(x, n=1, dim=2, append=x[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x_diff2 = torch.diff(x_diff1, n=1, dim=2, append=x_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x = torch.cat((x, x_diff1, x_diff2), dim=1)
        logmel_x = x

        if ifplot:
            x_array = x.data.cpu().numpy()[0][0]
            x_array = np.squeeze(x_array)
            plt.matshow(x_array.T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig('{}.jpg'.format(audio_name))

        x = self.cnn_encoder(x)

        # (samples_num, 512, hidden_units)
        output = F.max_pool2d(x, kernel_size=x.shape[2:])
        output = output.view(output.shape[0:2])
        # for tsne, features before the final fc layer
        output_tsne = output
        output = F.log_softmax(self.fc_final(output), dim=-1)

        return logmel_x, output_tsne, output


class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()

        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        # (batch_size, 3, time_steps, mel_bins)
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # (samples_num, channel, time_steps, freq_bins)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)

        return x


class Res_encoder(nn.Module):
    def __init__(self):
        super(Res_encoder, self).__init__()

        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ResBlock(in_channels=3, out_channels=64, stride=(1,1))
        self.conv2 = ResBlock(in_channels=64, out_channels=128, stride=(1,1))
        self.conv3 = ResBlock(in_channels=128, out_channels=256, stride=(1,1))
        self.conv4 = ResBlock(in_channels=256, out_channels=512, stride=(1,1))

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        # (batch_size, 3, time_steps, mel_bins)
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # (samples_num, channel, time_steps, freq_bins)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)

        return x