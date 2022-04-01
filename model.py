import torch
from torch import nn

from params import S, B, C, architecture_size, dropout

if architecture_size == 'full':
    architecture = [
        (7, 64, 2, 3),  # size, filters, stride, padding
        "M",  # Maxpool
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # [..., repeats]
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
    ]
    dense_size = 4096
elif architecture_size == 'mini':
    architecture = [
        (7, 64, 2, 3),
        "M",
        (3, 192, 1, 1),
        "M",
        (3, 256, 1, 1),
        (3, 512, 1, 1),
        "M",
        (3, 512, 1, 1),
        "M",
        (3, 512, 1, 1),
        (3, 512, 2, 1),
        (3, 512, 1, 1),
        (3, 1024, 1, 1)
    ]
    dense_size = 512
elif architecture_size == 'semi-mini':
    architecture = [
        (7, 64, 2, 3),
        "M",
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
    ]
    dense_size = 512


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0.1)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.1)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolo(nn.Module):
    """
    YOLO CNN model. Starts with Darknet architecture and finishes with FC layer. Though the original paper doesn't mention it, this model ends by applying softmax on class probabilities to improve training.

    Input is images with values between 0 and 1. Output is has shape (batch_size, S, S, C + B * 5) containing bounding box information for each grid cell.
    """

    def __init__(self, in_channels=3, **kwargs):
        super(Yolo, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        self.softmax = torch.nn.Softmax(dim=3)
    
    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        x = x.reshape(-1, S, S, C + B * 5)
        x[:, :, :, :C] = self.softmax(x[:, :, :, :C])
        return x
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(CNNBlock(
                    in_channels,
                    kernel_size=x[0],
                    out_channels=x[1],
                    stride=x[2],
                    padding=x[3]
                ))

                in_channels = x[1]
            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers.append(CNNBlock(
                        in_channels,
                        kernel_size=conv1[0],
                        out_channels=conv1[1],
                        stride=conv1[2],
                        padding=conv1[3]
                    ))
                    layers.append(CNNBlock(
                        conv1[1],
                        kernel_size=conv2[0],
                        out_channels=conv2[1],
                        stride=conv2[2],
                        padding=conv2[3]
                    ))

                    in_channels = conv2[1]

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def _create_fcs(self):
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, dense_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(dense_size, S * S * (C + B * 5))
        )
        net.apply(init_weights)
        return net
