import torch
from torch import nn

import utils


class AlexNetEncoder(nn.Module):
    def __init__(self, in_channel, feature_dim):
        super(AlexNetEncoder, self).__init__()
        self._conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self._conv_block_2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self._conv_block_3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self._conv_block_4 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self._conv_block_5 = nn.Sequential(
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self._encoder = nn.Sequential(
            nn.Linear(96 * 7 * 7, feature_dim)
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = obs / 255.0 - 0.5
        h = self._conv_block_1(h)
        h = self._conv_block_2(h)
        h = self._conv_block_3(h)
        h = self._conv_block_4(h)
        h = self._conv_block_5(h)
        h = h.view(h.shape[0], -1)
        h = self._encoder(h)
        return h


class ConvEncoder(nn.Module):
    def __init__(self, in_channel, feature_dim):
        super().__init__()
        self._conv_net = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, stride=2),  # 64 -> 31
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),  # 31 -> 29
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),  # 29 -> 27
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),  # 27 -> 25
            nn.ReLU()
        )
        self._encoder = nn.Sequential(
            nn.Linear(32 * 25 * 25, feature_dim)
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = obs / 255.0 - 0.5
        h = self._conv_net(h)
        h = h.view(h.shape[0], -1)
        h = self._encoder(h)
        return h


class LDDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, ld_hidden_dim):
        super(LDDecoder, self).__init__()
        self._encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, ld_hidden_dim),
            nn.ReLU(),

            nn.Linear(ld_hidden_dim, feature_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, states, actions):
        h = torch.cat([states, actions], dim=1)
        return self._encoder(h)


class LD(nn.Module):
    def __init__(self, nb_mlp, state_dim, action_dim, feature_dim, ld_hidden_dim):
        super(LD, self).__init__()
        self.nb_mlp = nb_mlp
        self._encoders = nn.ModuleList(
            [LDDecoder(state_dim, action_dim, feature_dim, ld_hidden_dim) for _ in range(nb_mlp)]
        )

    def forward(self, states, actions, mean=True):
        features = torch.stack([encoder(states, actions) for encoder in self._encoders], dim=1)
        feature_means = features.mean(dim=1)
        feature_var = features.var(dim=1).sum(dim=1)

        if mean:
            return feature_means, feature_var
        else:
            return torch.flatten(features, start_dim=1), feature_var


class StateEncoder(nn.Module):
    def __init__(self, feature_dim, state_dim):
        super(StateEncoder, self).__init__()
        self._encoder = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.Sigmoid(),
        )
        self.apply(utils.weight_init)

    def forward(self, features):
        return self._encoder(features)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self._evaluator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, states, actions):
        values = torch.cat([states, actions], dim=1)
        return self._evaluator(values)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self._policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(utils.weight_init)

    def forward(self, states, std):
        mu = self._policy(states)
        mu = torch.sigmoid(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

