from torch import nn

import utils


class HalfAlexNet(nn.Module):
    def __init__(self, in_channel):
        super(HalfAlexNet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.repr_dim = 96 * 7 * 7

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = obs / 255.0 - 0.5
        h = self.conv_block_1(h)
        h = self.conv_block_2(h)
        h = self.conv_block_3(h)
        h = self.conv_block_4(h)
        h = self.conv_block_5(h)
        h = h.view(h.shape[0], -1)
        return h


class ConvEncoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.repr_dim = 32 * 25 * 25

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, stride=2),  # 64 -> 31
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),  # 31 -> 29
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),  # 29 -> 27
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),  # 27 -> 25
            nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = obs / 255.0 - 0.5
        h = self.conv_net(h)
        h = h.view(h.shape[0], -1)
        return h

