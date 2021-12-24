import numpy as np
import torch.optim
from torch.nn import functional as F

import nets
import utils


class Explorer:

    def __init__(self, obs_channel, action_dim, lr, update_every_steps, use_tb, num_expl_steps, nb_ld_mlp, state_dim, hidden_dim, ld_hidden_dim, feature_dim, stddev_schedule, stddev_clip, nstep, discount):
        self.nb_ld_mlp = nb_ld_mlp
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.nstep = nstep
        self.discount = discount
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps

        self.encoder = nets.ConvEncoder(obs_channel, feature_dim).to(utils.device())
        self.state_encoder = nets.StateEncoder(feature_dim, state_dim).to(utils.device())
        self.actor = nets.Actor(state_dim, action_dim, hidden_dim).to(utils.device())
        self.ld = nets.LD(nb_ld_mlp, state_dim, action_dim, feature_dim, ld_hidden_dim).to(utils.device())

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr)
        self.state_encoder_opt = torch.optim.Adam(self.state_encoder.parameters(), lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.ld_opt = torch.optim.Adam(self.ld.parameters(), lr)

        self.train()

    def train(self, training=True):
        self.training = training

        self.encoder.train(training)
        self.state_encoder.train(training)
        self.actor.train(training)
        self.ld.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=utils.device())
        features = self.encoder(obs.unsqueeze(0))
        states = self.state_encoder(features)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(states, stddev)

        if eval_mode:
            actions = dist.mean
        else:
            actions = dist.sample(clip=None)
            if step < self.num_expl_steps:
                actions.uniform_(0., 1.)

        next_features, rewards = self.ld(states, actions, mean=True)

        return actions.cpu().numpy()[0], rewards.cpu().numpy()[0], next_features.cpu().numpy()[0]

    def update_ld(self, states, actions, next_features):
        metrics = dict()

        with torch.no_grad():
            next_features = next_features.repeat(1, self.nb_ld_mlp)
        pred_features, _ = self.ld(states, actions, mean=False)

        ld_loss = F.mse_loss(pred_features, next_features)

        if self.use_tb:
            metrics['ld_loss'] = ld_loss

        self.encoder_opt.zero_grad()
        self.state_encoder_opt.zero_grad()
        self.ld_opt.zero_grad()
        ld_loss.backward()
        self.ld_opt.step()
        self.state_encoder_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, states, step):
        metrics = dict()

        total_rewards = np.zeros(states.shape[0])
        discount = 1.

        for _ in range(self.nstep):
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(states, stddev)
            actions = dist.sample(clip=self.stddev_clip)

            next_features, rewards = self.ld(states, actions, mean=True)
            total_rewards += rewards * discount

            discount *= self.discount
            states = self.state_encoder(next_features)

        actor_loss = - total_rewards.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, actions, rewards, discounts, next_obs = utils.to_torch(batch)

        features = self.encoder(obs)
        states = self.state_encoder(features)
        with torch.no_grad():
            next_features = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = rewards.mean().item()

        metrics.update(self.update_ld(states, actions, next_features))

        metrics.update(self.update_actor(states.detach(), step))

        return metrics

