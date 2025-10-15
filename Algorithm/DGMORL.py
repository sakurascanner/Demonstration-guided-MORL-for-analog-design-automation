"""GPI-PD algorithm."""
import os
import random
from itertools import chain
from typing import List, Optional, Union
import mo_gymnasium as mo_gym
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from line_profiler import LineProfiler
from common.buffer import ReplayBuffer
from common.morl_algorithm import MOAgent, MOPolicy
from common.networks import (
    NatureCNN,
    get_grad_norm,
    huber,
    layer_init,
    mlp,
    polyak_update,
)
from common.prioritized_buffer import PrioritizedReplayBuffer
from common.utils import linearly_decaying_value, unique_tol
from common.weights import equally_spaced_weights
from linear_support import LinearSupport
from util.dataclass import demo_info

profiler = LineProfiler()


class QNet(nn.Module):
    """Conditioned MO Q network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch, drop_rate=0.01, layer_norm=True):
        """Initialize the net.

        Args:
            obs_shape: The observation shape.
            action_dim: The action dimension.
            rew_dim: The reward dimension.
            net_arch: The network architecture.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.phi_dim = rew_dim

        self.weights_features = mlp(rew_dim, -1, net_arch[:1])
        if len(obs_shape) == 1:
            self.state_features = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.state_features = NatureCNN(self.obs_shape, features_dim=net_arch[0])
        self.net = mlp(
            net_arch[0], action_dim * rew_dim, net_arch[1:], drop_rate=drop_rate, layer_norm=layer_norm
        )  # 128/128 256 256 256

        self.apply(layer_init)

    def forward(self, obs, w):
        """Forward pass."""
        sf = self.state_features(obs)
        wf = self.weights_features(w)
        q_values = self.net(sf * wf)
        return q_values.view(-1, self.action_dim, self.phi_dim)  # Batch size X Actions X Rewards


class GPIPD(MOPolicy, MOAgent):
    """GPI-PD Algorithm.

    Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization
    Lucas N. Alegre, Ana L. C. Bazzan, Diederik M. Roijers, Ann Nowé, Bruno C. da Silva
    AAMAS 2023
    Paper: https://arxiv.org/abs/2301.07784
    """

    def __init__(
            self,
            env,
            learning_rate: float = 3e-4,
            initial_epsilon: float = 0.01,
            final_epsilon: float = 0.01,
            epsilon_decay_steps: int = None,  # None == fixed epsilon
            tau: float = 1.0,
            target_net_update_freq: int = 1000,  # ignored if tau != 1.0
            buffer_size: int = int(1e6),
            net_arch: List = [256, 256, 256, 256],
            num_nets: int = 2,
            batch_size: int = 128,
            learning_starts: int = 100,
            gradient_updates: int = 20,
            gamma: float = 0.99,
            max_grad_norm: Optional[float] = None,
            use_gpi: bool = False,
            per: bool = True,
            gpi_pd: bool = False,
            alpha_per: float = 0.6,
            min_priority: float = 0.01,
            drop_rate: float = 0.01,
            layer_norm: bool = True,
            project_name: str = "MORL-Baselines",
            experiment_name: str = "GPI-PD",
            wandb_entity: Optional[str] = None,
            log: bool = True,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            eval_iterations=10,
            self_evolution=True
    ):
        """Initialize the GPI-PD algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            tau: The soft update coefficient.
            target_net_update_freq: The target network update frequency.
            buffer_size: The size of the replay buffer.
            net_arch: The network architecture.
            num_nets: The number of networks.
            batch_size: The batch size.
            learning_starts: The number of steps before learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor.
            max_grad_norm: The maximum gradient norm.
            use_gpi: Whether to use GPI.
            per: Whether to use PER.
            gpi_pd: Whether to use GPI-PD.
            alpha_per: The alpha parameter for PER.
            min_priority: The minimum priority for PER.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
            project_name: The name of the project.
            experiment_name: The name of the experiment.
            wandb_entity: The name of the wandb entity.
            log: Whether to log.
            seed: The seed for random number generators.
            device: The device to use.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.use_gpi = use_gpi
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.num_nets = num_nets
        self.drop_rate = drop_rate
        self.layer_norm = layer_norm
        self.mean_utility = []
        # Q-Networks
        self.q_nets = [
            QNet(
                self.observation_shape,
                self.action_dim,
                self.reward_dim,
                net_arch=net_arch,
                drop_rate=drop_rate,
                layer_norm=layer_norm,
            ).to(self.device)
            for _ in range(self.num_nets)
        ]
        self.target_q_nets = [
            QNet(
                self.observation_shape,
                self.action_dim,
                self.reward_dim,
                net_arch=net_arch,
                drop_rate=drop_rate,
                layer_norm=layer_norm,
            ).to(self.device)
            for _ in range(self.num_nets)
        ]
        for q, target_q in zip(self.q_nets, self.target_q_nets):
            target_q.load_state_dict(q.state_dict())
            for param in target_q.parameters():
                param.requires_grad = False
        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)

        # Prioritized experience replay parameters
        self.per = per
        self.gpi_pd = gpi_pd
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        self.min_priority = min_priority
        self.alpha = alpha_per

        # model-based parameters
        self.EU_list = []
        # logging
        self.log = log
        self.eval_iterations = eval_iterations
        self.linear_support = LinearSupport(num_objectives=self.reward_dim,
                                            epsilon=0.0)
        self.demo_repository = []
        self.self_evolution = self_evolution
        self.experiment_name = experiment_name
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Return the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "per": self.per,
            "gpi_pd": self.gpi_pd,
            "alpha_per": self.alpha,
            "min_priority": self.min_priority,
            "tau": self.tau,
            "num_nets": self.num_nets,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "drop_rate": self.drop_rate,
            "layer_norm": self.layer_norm,
            "seed": self.seed,
        }

    def save(self, save_replay_buffer=True, save_dir="weights/", filename=None):
        """Save the model parameters and the replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        for i, psi_net in enumerate(self.q_nets):
            saved_params[f"psi_net_{i}_state_dict"] = psi_net.state_dict()
        saved_params["psi_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        saved_params["M"] = self.weight_support
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the model parameters and the replay buffer."""
        params = th.load(path, map_location=self.device)
        for i, (psi_net, target_psi_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            psi_net.load_state_dict(params[f"psi_net_{i}_state_dict"])
            target_psi_net.load_state_dict(params[f"psi_net_{i}_state_dict"])
        self.q_optim.load_state_dict(params["psi_nets_optimizer_state_dict"])
        self.weight_support = params["M"]
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self, weight: th.Tensor):
        """Update the parameters of the networks."""
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self._sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self._sample_batch_experiences()
            if len(self.weight_support) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = (
                    s_obs.repeat(2, *(1 for _ in range(s_obs.dim() - 1))),
                    s_actions.repeat(2, 1),
                    s_rewards.repeat(2, 1),
                    s_next_obs.repeat(2, *(1 for _ in range(s_obs.dim() - 1))),
                    s_dones.repeat(2, 1),
                )
                # Half of the batch uses the given weight vector, the other half uses weights sampled from the support
                w = th.vstack(
                    [weight for _ in range(s_obs.size(0) // 2)] + random.choices(self.weight_support,
                                                                                 k=s_obs.size(0) // 2)
                )
            else:
                w = weight.repeat(s_obs.size(0), 1)

            if len(self.weight_support) > 5:
                sampled_w = th.stack([weight] + random.sample(self.weight_support, k=4))
            else:
                sampled_w = th.stack(self.weight_support)

            with th.no_grad():
                # Compute min_i Q_i(s', a, w) . w
                next_q_values = th.stack([target_psi_net(s_next_obs, w) for target_psi_net in self.target_q_nets])
                scalarized_next_q_values = th.einsum("nbar,br->nba", next_q_values, w)  # q_i(s', a, w)
                min_inds = th.argmin(scalarized_next_q_values, dim=0)
                min_inds = min_inds.reshape(1, next_q_values.size(1), next_q_values.size(2), 1).expand(
                    1, next_q_values.size(1), next_q_values.size(2), next_q_values.size(3)
                )
                next_q_values = next_q_values.gather(0, min_inds).squeeze(0)

                # Compute max_a Q(s', a, w) . w
                max_q = th.einsum("br,bar->ba", w, next_q_values)
                max_acts = th.argmax(max_q, dim=1)

                q_targets = next_q_values.gather(
                    1, max_acts.long().reshape(-1, 1, 1).expand(next_q_values.size(0), 1, next_q_values.size(2))
                )
                target_q = q_targets.reshape(-1, self.reward_dim)
                target_q = s_rewards + (1 - s_dones) * self.gamma * target_q

                if self.gpi_pd:
                    target_q_envelope, _ = self._envelope_target(s_next_obs, w, sampled_w)
                    target_q_envelope = s_rewards + (1 - s_dones) * self.gamma * target_q_envelope

            losses = []
            td_errors = []
            gtd_errors = []
            for psi_net in self.q_nets:
                psi_value = psi_net(s_obs, w)
                psi_value = psi_value.gather(
                    1, s_actions.long().reshape(-1, 1, 1).expand(psi_value.size(0), 1, psi_value.size(2))
                )
                psi_value = psi_value.reshape(-1, self.reward_dim)

                if self.gpi_pd:
                    gtd_error = psi_value - target_q_envelope

                td_error = psi_value - target_q
                loss = huber(td_error.abs(), min_priority=self.min_priority)
                losses.append(loss)
                if self.gpi_pd:
                    gtd_errors.append(gtd_error.abs())
                if self.per:
                    td_errors.append(td_error.abs())
            critic_loss = (1 / self.num_nets) * sum(losses)

            self.q_optim.zero_grad()
            critic_loss.backward()

            if self.max_grad_norm is not None:
                if self.log and self.global_step % 100 == 0:
                    wandb.log(
                        {
                            "losses/grad_norm": get_grad_norm(self.q_nets[0].parameters()).item(),
                            "global_step": self.global_step,
                        },
                    )
                for psi_net in self.q_nets:
                    th.nn.utils.clip_grad_norm_(psi_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per or self.gpi_pd:
                if self.gpi_pd:
                    gtd_error = th.max(th.stack(gtd_errors), dim=0)[0]
                    gtd_error = gtd_error[: len(idxes)].detach()
                    gper = th.einsum("br,br->b", w[: len(idxes)], gtd_error).abs()
                    gpriority = gper.cpu().numpy().flatten()
                    gpriority = gpriority.clip(min=self.min_priority) ** self.alpha

                if self.per:
                    td_error = th.max(th.stack(td_errors), dim=0)[0]
                    td_error = td_error[: len(idxes)].detach()
                    per = th.einsum("br,br->b", w[: len(idxes)], td_error).abs()
                    priority = per.cpu().numpy().flatten()
                    priority = priority.clip(min=self.min_priority) ** self.alpha

                if self.gpi_pd:
                    self.replay_buffer.update_priorities(idxes, gpriority)
                else:
                    self.replay_buffer.update_priorities(idxes, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            for psi_net, target_psi_net in zip(self.q_nets, self.target_q_nets):
                polyak_update(psi_net.parameters(), target_psi_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon, self.epsilon_decay_steps, self.global_step, self.learning_starts,
                self.final_epsilon
            )

    @th.no_grad()
    def gpi_action(self, obs: th.Tensor, w: th.Tensor, return_policy_index=False, include_w=False):
        """Select an action using GPI."""
        if include_w:
            M = th.stack(self.weight_support + [w])
        else:
            M = th.stack(self.weight_support)

        obs_m = obs.repeat(M.size(0), *(1 for _ in range(obs.dim())))
        q_values = self.q_nets[0](obs_m, M)

        scalar_q_values = th.einsum("r,bar->ba", w, q_values)  # q(s,a,w_i) = q(s,a,w_i) . w
        max_q, a = th.max(scalar_q_values, dim=1)
        policy_index = th.argmax(max_q)  # max_i max_a q(s,a,w_i)
        action = a[policy_index].detach().item()

        if return_policy_index:
            return action, policy_index.item()
        return action

    @th.no_grad()
    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        """Select an action for the given obs and weight vector."""
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        if self.use_gpi:
            action = self.gpi_action(obs, w, include_w=False)
        else:
            action = self.max_action(obs, w)
        return action

    def _act(self, obs: th.Tensor, w: th.Tensor) -> int:
        if self.np_random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.use_gpi:
                action, policy_index = self.gpi_action(obs, w, return_policy_index=True)
                self.police_indices.append(policy_index)
                return action
            else:
                return self.max_action(obs, w)

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Select the greedy action."""
        psi = th.min(th.stack([psi_net(obs, w) for psi_net in self.q_nets]), dim=0)[0]
        # psi = self.psi_nets[0](obs, w)
        q = th.einsum("r,bar->ba", w, psi)
        max_act = th.argmax(q, dim=1)
        return max_act.detach().item()

    @th.no_grad()
    def _reset_priorities(self, w: th.Tensor):
        inds = np.arange(self.replay_buffer.size)
        priorities = np.repeat(0.1, self.replay_buffer.size)
        (
            obs_s,
            actions_s,
            rewards_s,
            next_obs_s,
            dones_s,
        ) = self.replay_buffer.get_all_data(to_tensor=False)
        num_batches = int(np.ceil(obs_s.shape[0] / 1000))
        for i in range(num_batches):
            b = i * 1000
            e = min((i + 1) * 1000, obs_s.shape[0])
            obs, actions, rewards, next_obs, dones = obs_s[b:e], actions_s[b:e], rewards_s[b:e], next_obs_s[
                                                                                                 b:e], dones_s[b:e]
            obs, actions, rewards, next_obs, dones = (
                th.tensor(obs).to(self.device),
                th.tensor(actions).to(self.device),
                th.tensor(rewards).to(self.device),
                th.tensor(next_obs).to(self.device),
                th.tensor(dones).to(self.device),
            )
            q_values = self.q_nets[0](obs, w.repeat(obs.size(0), 1))
            q_a = q_values.gather(1, actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1,
                                                                             q_values.size(2))).squeeze(1)

            if self.gpi_pd:
                max_next_q, _ = self._envelope_target(next_obs, w.repeat(next_obs.size(0), 1),
                                                      th.stack(self.weight_support))
            else:
                next_q_values = self.q_nets[0](next_obs, w.repeat(next_obs.size(0), 1))
                max_q = th.einsum("r,bar->ba", w, next_q_values)
                max_acts = th.argmax(max_q, dim=1)
                q_targets = self.target_q_nets[0](next_obs, w.repeat(next_obs.size(0), 1))
                q_targets = q_targets.gather(
                    1, max_acts.long().reshape(-1, 1, 1).expand(q_targets.size(0), 1, q_targets.size(2))
                )
                max_next_q = q_targets.reshape(-1, self.reward_dim)

            gtderror = th.einsum("r,br->b", w, (rewards + (1 - dones) * self.gamma * max_next_q - q_a)).abs()
            priorities[b:e] = gtderror.clamp(min=self.min_priority).pow(self.alpha).cpu().detach().numpy().flatten()

        self.replay_buffer.update_priorities(inds, priorities)

    @th.no_grad()
    def _envelope_target(self, obs: th.Tensor, w: th.Tensor, sampled_w: th.Tensor):
        W = sampled_w.unsqueeze(0).repeat(obs.size(0), 1, 1)
        next_obs = obs.unsqueeze(1).repeat(1, sampled_w.size(0), 1)

        next_q_target = th.stack(
            [
                target_net(next_obs, W).view(obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
                for target_net in self.target_q_nets
            ]
        )

        q_values = th.einsum("br,nbpar->nbpa", w, next_q_target)
        min_inds = th.argmin(q_values, dim=0)
        min_inds = min_inds.reshape(1, next_q_target.size(1), next_q_target.size(2), next_q_target.size(3), 1).expand(
            1, next_q_target.size(1), next_q_target.size(2), next_q_target.size(3), next_q_target.size(4)
        )
        next_q_target = next_q_target.gather(0, min_inds).squeeze(0)

        q_values = th.einsum("br,bpar->bpa", w, next_q_target)
        max_q, ac = th.max(q_values, dim=2)
        pi = th.argmax(max_q, dim=1)

        max_next_q = next_q_target.gather(
            2,
            ac.unsqueeze(2).unsqueeze(3).expand(next_q_target.size(0), next_q_target.size(1), 1, next_q_target.size(3)),
        ).squeeze(2)
        max_next_q = max_next_q.gather(1,
                                       pi.reshape(-1, 1, 1).expand(max_next_q.size(0), 1, max_next_q.size(2))).squeeze(
            1)
        return max_next_q, next_q_target

    def set_weight_support(self, weight_list: List[np.ndarray]):
        """Set the weight support set."""
        weights_no_repeats = unique_tol(weight_list)
        self.weight_support = [th.tensor(w).float().to(self.device) for w in weights_no_repeats]

    def add_new_demo_info(self, new_demo, disc_vec_return, horizon):
        if horizon is None:
            d_info = demo_info(demo=new_demo, disc_vec_return=disc_vec_return, updated=False, horizon=len(new_demo))
        else:
            d_info = demo_info(demo=new_demo, disc_vec_return=disc_vec_return, updated=False, horizon=horizon)
        self.demo_repository.append(d_info)
        return d_info

    def update_linear_support_ccs(self):
        ccs = []
        print("--update LS ccs")
        for demo_info in self.demo_repository:
            # print(f"demo:{demo_info.demo} - vec_return {np.round_(demo_info.disc_vec_return, 4)} added")
            ccs.append(demo_info.disc_vec_return)
        self.linear_support.ccs = ccs
        return self.linear_support

    def renew_horizon(self):
        for demo_info in self.demo_repository:
            demo_info.horizon = len(demo_info.demo)

    def remove_obsolete_demos(self):
        for i in reversed(range(len(self.demo_repository))):
            updated = self.demo_repository[i].updated
            is_dominated = self.linear_support.is_dominated(self.demo_repository[i].disc_vec_return)
            if updated and is_dominated:
                print(
                    f"demo:{self.demo_repository[i].demo} - vec:{np.round_(self.demo_repository[i].disc_vec_return, 4)} is removed")
                self.demo_repository.pop(i)

    def get_corners_(self):
        corners = self.linear_support.compute_corner_weights()
        for w in corners:
            self.linear_support.visited_weights.append(w)
        return corners

    def get_demos_EU_(self, eval_weights, eval_env):
        utility_thresholds = []
        for w_c in eval_weights:
            demo, utility_threshold, vec_return, _ = self.weight_to_demo_(w=w_c, eval_env=eval_env,
                                                                          eval_iterations=self.eval_iterations)
            # print(f"vec_returns:{vec_return}")
            utility_thresholds.append(utility_threshold)
        utility_thresholds = np.array(utility_thresholds)

        EU_target = np.mean(utility_thresholds)
        # print(f"EU-target:{EU_target}")
        return EU_target

    def get_agent_EU_(self, eval_env, eval_weights, eval_iterations, show_case=True, path=None, info=None):
        utilities = []
        demos = []
        for weight in eval_weights:
            disc_return, disc_vec_return, scalar_return, vec_return, new_demo = self.play_a_episode_(env=eval_env,
                                                                                                     weight=weight,
                                                                                                     demo_info=demo_info(
                                                                                                         horizon=0),
                                                                                                     eval_iterations=eval_iterations)
            utilities.append(disc_return)
            if show_case:
                print(f"for w:{np.round_(weight, 3)}\tdisc vec return:{vec_return}")
                with open(path + '/disc_vec_ret_' + info + '.txt', 'a') as file:
                    print(f"for w:{np.round_(weight, 3)}\tdisc vec return:{disc_vec_return}\n", file=file)
            demos.append(new_demo)
        EU = np.mean(utilities)
        if show_case:
            print(f"EU:{EU}")
        return EU, demos

    def weight_to_demo_(self, w, eval_env, eval_iterations):
        disc_scalar_returns = []
        disc_vec_returns = []
        for demo_info in self.demo_repository:
            utility, disc_vec_return, scalar_return, vec_return = self.evaluate_demo_(demo=demo_info.demo,
                                                                                      eval_env=eval_env, weights=w,
                                                                                      eval_iterations=eval_iterations,
                                                                                      mode="min")
            disc_scalar_returns.append(utility)
            disc_vec_returns.append(disc_vec_return)
        max_demo_idx = np.argmax(disc_scalar_returns)
        max_demo_info = self.demo_repository[max_demo_idx]
        max_utility = disc_scalar_returns[max_demo_idx]
        max_vec_return = disc_vec_returns[max_demo_idx]
        return max_demo_info, max_utility, max_vec_return, max_demo_idx

    def play_a_episode_(self, env, weight, eval_iterations, demo_info: demo_info, evaluation=False, max_steps=100):
        epsilon = self.epsilon

        terminated = False
        truncated = False

        new_demo = []
        action_pointer = 0
        steps = 0
        self.epsilon = 0

        obs, _ = env.reset()
        while not terminated and not truncated:
            steps += 1
            if action_pointer < demo_info.horizon:
                action = demo_info.demo[action_pointer]
                action_pointer += 1
            else:
                action = self._act(th.as_tensor(obs).float().to(self.device),
                                   th.as_tensor(weight).float().to(self.device))

            obs_, rewards, terminated, truncated, _ = env.step(action)
            obs = obs_
            new_demo.append(action)
            # if steps > max_steps:            #     break
        if evaluation:
            print(f"eval action traj:{new_demo}")
        self.epsilon = epsilon
        avg_utility, avg_disc_vec_return, avg_scalar_return, avg_vec_return = self.evaluate_demo_(new_demo,
                                                                                                  eval_env,
                                                                                                  weights=weight,
                                                                                                  eval_iterations=eval_iterations)

        return avg_utility, avg_disc_vec_return, avg_scalar_return, avg_vec_return, new_demo

    def evaluate_demo_(self, demo, eval_env, weights=np.zeros([1, 0]), eval_iterations=5, mode="mean"):

        disc_scalar_returns = []
        scalar_returns = []
        disc_vec_returns = []
        vec_returns = []

        for _ in range(eval_iterations):
            gamma = 1
            disc_scalar_return = 0
            scalar_return = 0
            disc_vec_return = np.zeros(self.reward_dim, dtype=np.float64)
            vec_return = np.zeros(self.reward_dim, dtype=np.float64)
            obs, _ = eval_env.reset()
            for action in demo:
                _, rewards, terminated, truncated, _ = eval_env.step(action)
                disc_scalar_return += gamma * np.dot(rewards, weights)
                disc_vec_return += gamma * rewards

                scalar_return += np.dot(rewards, weights)
                vec_return += rewards

                gamma *= self.gamma
                if terminated or truncated:
                    break
            disc_scalar_returns.append(disc_scalar_return)
            scalar_returns.append(scalar_return)
            disc_vec_returns.append(disc_vec_return)
            vec_returns.append(vec_return)
        if mode == "mean":
            _disc_scalar_return = np.mean(disc_scalar_returns)
            _disc_vec_return = np.mean(disc_vec_returns, axis=0)
            _scalar_return = np.mean(scalar_returns)
            _vec_return = np.mean(vec_returns, axis=0)
        else:
            idx = np.argmax(disc_scalar_returns)
            _disc_scalar_return = np.min(disc_scalar_returns)
            _disc_vec_return = disc_vec_returns[idx]
            _scalar_return = np.mean(scalar_returns)
            _vec_return = vec_returns[idx]
        # print(f"_disc_scalar_return:{_disc_scalar_return}\t_disc_vec_return:{_disc_vec_return}\t_scalar_return:{_scalar_return}\t_vec_return:{_vec_return}")
        return _disc_scalar_return, _disc_vec_return, _scalar_return, _vec_return

    def select_w_(self, eval_env, weights):
        priorities = []
        demos_info_ = []
        u_thresholds = []
        max_disc_vec_returns = []
        for w in weights:
            demo_info, utility_threshold, max_disc_vec_return, demo_idx = self.weight_to_demo_(w=w, eval_env=eval_env,
                                                                                               eval_iterations=self.eval_iterations)
            u, _ = self.get_agent_EU_(eval_env=eval_env,
                                      eval_weights=np.array([w]),
                                      show_case=False,
                                      eval_iterations=self.eval_iterations)
            priority = abs((utility_threshold - u) / utility_threshold) + 1e-20
            priorities.append(priority)
            demos_info_.append(demo_info)
            u_thresholds.append(utility_threshold)
            max_disc_vec_returns.append(max_disc_vec_return)
        idx = random.choices(range(len(priorities)), weights=priorities, k=1)[0]
        w = weights[idx]
        demo_info = demos_info_[idx]
        utility_threshold = u_thresholds[idx]
        max_disc_vec_return = max_disc_vec_returns[idx]
        print(f"select:{w}by priority of:{priorities[idx]}")
        return w, demo_info, utility_threshold, max_disc_vec_return

    def jsmorl_train(self,
                     demos,
                     eval_env,
                     total_timesteps,
                     timesteps_per_iter,
                     title="",
                     eval_freq=100,
                     roll_back_step=2
                     ):

        for demo in demos:
            avg_utility, avg_disc_vec_return, avg_scalar_return, avg_vec_return = self.evaluate_demo_(demo,
                                                                                                      eval_env,
                                                                                                      weights=np.zeros(
                                                                                                          self.reward_dim),
                                                                                                      eval_iterations=self.eval_iterations,
                                                                                                      mode="min")
            print(f"demo vec:{avg_disc_vec_return}")
            self.add_new_demo_info(new_demo=demo, disc_vec_return=avg_disc_vec_return, horizon=None)
        self.update_linear_support_ccs()
        corners = self.get_corners_()
        print(f"corner weights:{np.round_(corners, 3)}")
        self.set_weight_support(corners)

        while self.global_step < total_timesteps:
            self.jsmorl_train_iteration(eval_env=eval_env,
                                        eval_freq=eval_freq,
                                        weight_support=corners,
                                        total_timesteps=timesteps_per_iter,
                                        roll_back_step=roll_back_step,
                                        path=title)

            self.update_linear_support_ccs()
            self.remove_obsolete_demos()
            for demo_info in self.demo_repository:
                print(f"@REPO --> demo:{demo_info.demo}\tvec_return {np.round_(demo_info.disc_vec_return, 4)}")
            self.renew_horizon()
            corners = self.get_corners_()
            self.set_weight_support(corners)

        print("----train_finish----")
        file_name = title + "/_15.npy"
        np.save(file_name, self.EU_list)
        print(f"saved")
        corners = self.get_corners_()
        print(f"corner weights:{np.round_(corners, 3)}")
        eval_weights = equally_spaced_weights(self.reward_dim, n=500)
        self.get_agent_EU_(eval_env=eval_env,
                           eval_weights=eval_weights,
                           eval_iterations=self.eval_iterations,
                           show_case=True)
        # plt.plot(self.EU_list)
        # plt.show()

    def jsmorl_train_iteration(self,
                               eval_env=None,
                               eval_freq: int = 100,
                               weight_support=None,
                               total_timesteps=4000,
                               roll_back_step=2,
                               path=None,
                               ):
        """Train the agent for one iteration.
                Args:
                    eval_env (Optional[gym.Env]): Environment to evaluate on
                    eval_freq (int): Number of timesteps between evaluations
        """
        self.police_indices = []
        eval_weights = equally_spaced_weights(self.reward_dim, n=500)
        candidate_ws = random.choices(eval_weights, k=len(weight_support))
        weight_support += candidate_ws

        w, demo_info, utility_threshold, disc_vec_return = self.select_w_(eval_env=eval_env,
                                                                          weights=weight_support)
        self.linear_support.visited_weights.append(w)
        tensor_w = th.tensor(w).float().to(self.device)

        pi_g_pointer = 0
        # demo_info.horizon -= roll_back_step
        obs, _ = self.env.reset()
        step = 0
        episode_steps = 0
        if self.per and len(self.replay_buffer) > 0:
            self._reset_priorities(tensor_w)
        departed = False

        # EU_target = self.get_demos_EU_(eval_weights=eval_weights, eval_env=eval_env)
        # print(f"@{self.seed}\tEU_target:{EU_target}")

        while step < total_timesteps and demo_info.horizon >= 0:
            step += 1
            episode_steps += 1
            self.global_step += 1
            if departed == False:
                if demo_info.horizon > 0 and pi_g_pointer < demo_info.horizon:
                    action = demo_info.demo[:demo_info.horizon][pi_g_pointer]
                    pi_g_pointer += 1
                else:
                    action = self._act(th.as_tensor(obs).float().to(self.device), tensor_w)
            else:
                action = self._act(th.as_tensor(obs).float().to(self.device), tensor_w)
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                if self.global_step == self.learning_starts:
                    print(">>>>>>>>>>>>>START TRAIN<<<<<<<<<<<<<<<")
                self.update(tensor_w)

            if self.global_step % eval_freq == 0:
                u, disc_vec_return, _, _, new_demo = self.play_a_episode_(env=eval_env,
                                                                          weight=w,
                                                                          demo_info=demo_info,
                                                                          evaluation=False,
                                                                          eval_iterations=self.eval_iterations)
                # u_thres_factor = min(0.9 + 0.1 * (self.global_step / 15e3), 1)
                if np.round_(u, 4) >= np.round_(utility_threshold, 4):
                    print(f"weight: {w} -- reach threshold - "
                          f"u:{np.round_(u, 4)}>u_thre:{np.round_(utility_threshold, 4)}")

                    if demo_info.horizon >= 1:
                        demo_info.horizon = max(demo_info.horizon - roll_back_step, 0)
                    else:
                        demo_info.horizon -= roll_back_step
                    if self.self_evolution:
                        if np.round_(u, 4) >= np.round_(utility_threshold, 4):
                            demo_info.updated = True
                            print(f"replace old demo {demo_info.demo} with better demo:{new_demo}")
                            demo_info = self.add_new_demo_info(new_demo=new_demo,
                                                               disc_vec_return=disc_vec_return,
                                                               horizon=demo_info.horizon)

                print(
                    f"@{self.global_step}\tguide policy horizon:{demo_info.horizon}\t"
                    f"demo:{new_demo}\t"
                    f"w_c:{w}\t"
                    f"u:{np.round_(u, 4)}\t"
                    f"u_threshold:{np.round_(utility_threshold, 4)}\t"
                    f"epsilon:{self.epsilon}")

            # if self.global_step % 50 == 0:
            #     print(f"@step:{self.global_step}")
            #     EU = self.get_agent_EU_(eval_weights=eval_weights, eval_env=eval_env,
            #                             eval_iterations=self.eval_iterations, show_case=True)
            #     EU_target = self.get_demos_EU_(eval_weights=eval_weights, eval_env=eval_env)

            if self.global_step % total_timesteps == 0:
                EU, new_demos = self.get_agent_EU_(eval_weights=eval_weights, eval_env=eval_env,
                                                   eval_iterations=self.eval_iterations, show_case=True,
                                                   path=path,
                                                   info=str(self.global_step))
                dir_path = "Algorithm/" + self.experiment_name + "/action_seq/" + str(self.global_step) + "/"
                if not os.path.exists(dir_path):
                    print(f"make dir{dir_path}")
                    os.makedirs(dir_path)
                for i in range(len(new_demos)):
                    # demo_info = self.demo_repository[i]
                    np.save(dir_path + str(i), new_demos[i])
                # EU_target = self.get_demos_EU_(eval_weights=eval_weights, eval_env=eval_env)
                print(f"@{self.global_step}\tEU:{EU}"
                      # f"\tEU_target:{EU_target}"
                      f"")

                self.EU_list.append(EU)

            if terminated or truncated:
                if departed == False:
                    departed = True
                if departed == True:
                    departed = False
                self.police_indices = []
                episode_steps = 0
                pi_g_pointer = 0
                # w = random.choice(weight_support)
                # print(f"change to w:{w}")
                # demo_info, utility_threshold, _, _ = self.weight_to_demo_(w=w)
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        return self.demo_repository


class GPILS(GPIPD):
    """Model-free version of GPI-PD."""

    def __init__(self, *args, **kwargs):
        """Initialize GPI-LS deactivating the dynamics model."""
        super().__init__(gpi_pd=False, experiment_name="GPI-LS", *args, **kwargs)


if __name__ == '__main__':
    seed = 42
    nums_demo = [6]
    seeds = [
        2,
        7,
        15,
        42,
        78
    ]
    reps = [2, 2, 4]
    # roll_back_spans = [5, 3, 1]
    env_name = "minecart"
    experiment_type = "main"
    for seed in seeds:
        save_dir = "../results/" + env_name + "/test_free_passing" + experiment_type + "_seed_" + str(seed)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        """ This part is for Experiment of Minecart"""
        env = mo_gym.make("minecart-v0", max_episode_steps=30)
        env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.98)
        eval_env = mo_gym.make("minecart-v0", max_episode_steps=30)
        eval_env = mo_gym.MORecordEpisodeStatistics(eval_env, gamma=0.98)

        agent = GPIPD(
            env,
            num_nets=2,
            max_grad_norm=None,
            learning_rate=3e-4,
            gamma=0.98,
            batch_size=128,
            net_arch=[256, 256, 256, 256],
            buffer_size=int(2e5),
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=50000,
            learning_starts=100,
            alpha_per=0.6,
            min_priority=0.01,
            per=True,
            gpi_pd=False,
            use_gpi=False,
            target_net_update_freq=200,
            tau=1,
            log=False,
            project_name="MORL-Baselines",
            experiment_name="GPI-PD_minecart",
            eval_iterations=5,
            seed=seed,
            self_evolution=False
        )
        human_demos = np.load("../train/minecart/traj/demos.npy", allow_pickle=True)
        print(f"len human_demos:{len(human_demos)}")
        # human_demos = [
        #     human_demos[1],
        #     human_demos[2], human_demos[6]]
        # human_demos = random.choices(human_demos, k=2)
        print(f"human demos:{human_demos}")
        timesteps_per_iter = 10000
        agent.jsmorl_train(demos=human_demos, eval_env=eval_env, total_timesteps=15 * timesteps_per_iter,
                           timesteps_per_iter=timesteps_per_iter,
                           title=save_dir,
                           eval_freq=100,
                           roll_back_step=15)


# if __name__ == '__main__':
#     seed = 42
#     nums_demo = [6]
#     seeds = [
#         2,
#         7,
#         15,
#         42,
#         78
#     ]
#     reps = [2, 2, 4]
#     # roll_back_spans = [5, 3, 1]
#     env_name = "minecart"
#     experiment_type = "main"
#     for seed in seeds:
#         save_dir = "../results/" + env_name + "/test_free_passing" + experiment_type + "_seed_" + str(seed)
#         if not os.path.isdir(save_dir):
#             os.makedirs(save_dir)
#         # # print(f"roll back:{roll_back_span}>>")
#         # # for num_demo in nums_demo:
#         # env = mo_gym.make("deep-sea-treasure-v0")
#         # eval_env = mo_gym.make("deep-sea-treasure-v0")
#         #
#         # agent = GPIPD(
#         #     env,
#         #     num_nets=2,
#         #     max_grad_norm=None,
#         #     learning_rate=3e-4,
#         #     gamma=0.99,
#         #     batch_size=128,
#         #     net_arch=[256, 256, 256, 256],
#         #     buffer_size=int(2e5),
#         #     initial_epsilon=1.0,
#         #     final_epsilon=0.05,
#         #     epsilon_decay_steps=50000,
#         #     learning_starts=100,
#         #     alpha_per=0.6,
#         #     min_priority=0.01,
#         #     gpi_pd=False,
#         #     use_gpi=False,
#         #     target_net_update_freq=200,
#         #     tau=1,
#         #     log=False,
#         #     project_name="MORL-Baselines",
#         #     experiment_name="GPI-PD",
#         #     seed=78,
#         #     self_evolution=True
#         # )
#         # action_demo_1 = [2, 1]  # 0.7
#         # action_demo_2 = [2, 3, 1, 1]  # 8.2
#         # action_demo_3 = [2, 3, 3, 1, 1, 1]  # 11.5
#         # action_demo_4 = [2, 3, 3, 3, 1, 1, 1, 1]  # 14.0
#         # action_demo_5 = [2, 3, 3, 3, 3, 1, 1, 1, 1]  # good.1
#         # action_demo_6 = [2, 3, 3, 3, 3, 3, 1, 1, 1, 1]  # 16.1
#         # action_demo_7 = [2, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 19.6
#         # action_demo_8 = [2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 20.3
#         # action_demo_9 = [2, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 22.4
#         # action_demo_10 = [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 23.7
#         # # action_demo_1 = [ 1]  # 0.7
#         # # action_demo_2 = [3, 1, 1]  # 8.2
#         # # action_demo_3 = [3, 3, 1, 1, 1]  # 11.5
#         # # action_demo_4 = [3, 3, 3, 1, 1, 1, 1]  # 14.0
#         # # action_demo_5 = [3, 3, 3, 3, 1, 1, 1, 1]  # good.1
#         # # action_demo_6 = [3, 3, 3, 3, 3, 1, 1, 1, 1]  # 16.1
#         # # action_demo_7 = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 19.6
#         # # action_demo_8 = [3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 20.3
#         # # action_demo_9 = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 22.4
#         # # action_demo_10 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 23.7
#         #
#         #
#         # # action_demos = [action_demo_1, action_demo_2, action_demo_3, action_demo_4, action_demo_5, action_demo_6,
#         # #                 action_demo_7, action_demo_8, action_demo_9, action_demo_10]
#         # action_demos = [action_demo_1, action_demo_2, action_demo_3, action_demo_4, action_demo_5, action_demo_8,
#         #                 action_demo_9, action_demo_10]
#         # # action_demos = random.choices(action_demos, k=num_demo)
#         # print(f"action_demos:{action_demos}")
#         # # print(agent.play_a_episode(env=env, demo=[3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         # #                            weights=np.array([1, 0]), agent=agent))
#         # # print(agent.evaluate_demo(demo=[3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], eval_env=eval_env,
#         # #                           weights=np.array([1, 0])))
#         # eval_weights = equally_spaced_weights(2, n=100)
#         # # for w in eval_weights:
#         # #     print(w)
#         # # w = random.choices(eval_weights)
#         # # corners = agent.get_corners_(demos=action_demos)
#         # # print(f"corners:{corners}")
#         # agent.jsmorl_train(demos=action_demos, eval_env=eval_env, total_timesteps=40000, timesteps_per_iter=4000,
#         #                    # eval_freq=500,
#         #                    title="Rollback_2" +
#         #                          # str(roll_back_span) +
#         #                          "_seed_" + str(seed),
#         #                    roll_back_step=2)

#         """ This part is for Experiment of Minecart"""
#         env = mo_gym.make("minecart-v0", max_episode_steps=30)
#         env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.98)
#         eval_env = mo_gym.make("minecart-v0", max_episode_steps=30)
#         eval_env = mo_gym.MORecordEpisodeStatistics(eval_env, gamma=0.98)

#         agent = GPIPD(
#             env,
#             num_nets=2,
#             max_grad_norm=None,
#             learning_rate=3e-4,
#             gamma=0.98,
#             batch_size=128,
#             net_arch=[256, 256, 256, 256],
#             buffer_size=int(2e5),
#             initial_epsilon=1.0,
#             final_epsilon=0.05,
#             epsilon_decay_steps=50000,
#             learning_starts=100,
#             alpha_per=0.6,
#             min_priority=0.01,
#             per=True,
#             gpi_pd=False,
#             use_gpi=False,
#             target_net_update_freq=200,
#             tau=1,
#             log=False,
#             project_name="MORL-Baselines",
#             experiment_name="GPI-PD_minecart",
#             eval_iterations=5,
#             seed=seed,
#             self_evolution=False
#         )
#         human_demos = np.load("../train/minecart/traj/demos.npy", allow_pickle=True)
#         print(f"len human_demos:{len(human_demos)}")
#         # human_demos = [
#         #     human_demos[1],
#         #     human_demos[2], human_demos[6]]
#         # human_demos = random.choices(human_demos, k=2)
#         print(f"human demos:{human_demos}")
#         timesteps_per_iter = 10000
#         agent.jsmorl_train(demos=human_demos, eval_env=eval_env, total_timesteps=15 * timesteps_per_iter,
#                            timesteps_per_iter=timesteps_per_iter,
#                            title=save_dir,
#                            eval_freq=100,
#                            roll_back_step=15)
