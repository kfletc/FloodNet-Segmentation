# model.py
# contains classes defining the network model
# defines forward pass through network

from . import config

import torch
import torch.nn as nn
import torch.nn.functional as f

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')

    def forward(self, x):
        # foward pass through a block
        return f.relu(self.conv2(f.relu(self.conv1(x))))
class SegmentationNet(nn.Module):
    def __init__(self, num_classes=10, retain_dim=True, out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()

        self.retain_dim = retain_dim
        self.out_size = out_size

        # blocks of encoder convolutional layers
        self.enc_blocks = nn.ModuleList([Block(3, 16), Block(16, 32), Block(32, 64), Block(64, 128)])
        # blocks of decoder convolutional layers
        self.dec_blocks = nn.ModuleList([Block(128, 64), Block(64, 32), Block(32, 16)])

        # pooling layer
        self.pool = nn.MaxPool2d(2)
        # upsample layers
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(128, 64, 2, 2), nn.ConvTranspose2d(64, 32, 2, 2),
                                      nn.ConvTranspose2d(32, 16, 2, 2)])
        # final layer
        self.head = nn.Conv2d(16, num_classes, 1, padding='same')

    def forward(self, x):
        # foward pass of entire network
        encode_layer_1 = self.enc_blocks[0](x)
        after_pool = self.pool(encode_layer_1)
        encode_layer_2 = self.enc_blocks[1](after_pool)
        after_pool = self.pool(encode_layer_2)
        encode_layer_3 = self.enc_blocks[2](after_pool)
        after_pool = self.pool(encode_layer_3)
        encode_layer_4 = self.enc_blocks[3](after_pool)
        upsample = self.upconvs[0](encode_layer_4)
        skip_connection_1 = torch.cat([upsample, encode_layer_3], dim=1)
        decode_layer_1 = self.dec_blocks[0](skip_connection_1)
        upsample = self.upconvs[1](decode_layer_1)
        skip_connection_2 = torch.cat([upsample, encode_layer_2], dim=1)
        decode_layer_2 = self.dec_blocks[1](skip_connection_2)
        upsample = self.upconvs[2](decode_layer_2)
        skip_connection_3 = torch.cat([upsample, encode_layer_1], dim=1)
        decode_layer_3 = self.dec_blocks[2](skip_connection_3)
        output_map = self.head(decode_layer_3)

        # for sanity resizes to input size
        if self.retain_dim:
            output_map = f.interpolate(output_map, self.out_size)

        return output_map
