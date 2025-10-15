"""GPI-PD algorithm with continuous actions."""
import copy
import os
import random
from itertools import chain
from typing import List, Optional, Union
import mo_gymnasium as mo_gym
import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import time
from matplotlib import pyplot as plt
from common.buffer import ReplayBuffer
from common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from common.model_based.probabilistic_ensemble import (
    ProbabilisticEnsemble,
)
from common.model_based.utils import ModelEnv, visualize_eval
from common.morl_algorithm import MOAgent, MOPolicy
from common.networks import layer_init, mlp, polyak_update
from common.prioritized_buffer import PrioritizedReplayBuffer
from common.utils import unique_tol
from common.weights import equally_spaced_weights
from linear_support import LinearSupport
from util.dataclass import demo_info
import time


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_


class Policy(nn.Module):
    """Policy network."""

    def __init__(self, obs_dim, rew_dim, output_dim, action_space, net_arch=[256, 256]):
        """Initialize the policy network."""
        super().__init__()
        self.action_space = action_space
        self.latent_pi = mlp(obs_dim + rew_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], output_dim)

        # action rescaling
        self.register_buffer("action_scale", th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32))

        self.apply(layer_init)

    def forward(self, obs, w, noise=None, noise_clip=None):
        """Forward pass of the policy network."""
        h = self.latent_pi(th.concat((obs, w), dim=obs.dim() - 1))
        action = self.mean(h)
        action = th.tanh(action)
        if noise is not None:
            n = (th.randn_like(action) * noise).clamp(-noise_clip, noise_clip)
            action = (action + n).clamp(-1, 1)
        return action * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """Q-network S x Ax W -> R^reward_dim."""

    def __init__(self, obs_dim, action_dim, rew_dim, net_arch=[256, 256], layer_norm=True, drop_rate=0.01):
        """Initialize the Q-network."""
        super().__init__()
        self.net = mlp(obs_dim + action_dim + rew_dim, rew_dim, net_arch, drop_rate=drop_rate, layer_norm=layer_norm)
        self.apply(layer_init)

    def forward(self, obs, action, w):
        """Forward pass of the Q-network."""
        q_values = self.net(th.cat((obs, action, w), dim=obs.dim() - 1))
        return q_values


class GPIPDContinuousAction(MOAgent, MOPolicy):
    """GPI-PD algorithm with continuous actions.

    Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization
    Lucas N. Alegre, Ana L. C. Bazzan, Diederik M. Roijers, Ann Nowé, Bruno C. da Silva
    AAMAS 2023
    Paper: https://arxiv.org/abs/2301.07784
    See Appendix for Continuous Action details.
    """

    def __init__(
            self,
            env,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            buffer_size: int = 400000,
            net_arch: List = [256, 256],
            batch_size: int = 128,
            num_q_nets: int = 2,
            delay_policy_update: int = 2,
            learning_starts: int = 100,
            gradient_updates: int = 1,
            use_gpi: bool = False,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            per: bool = True,
            min_priority: float = 0.1,
            alpha: float = 0.6,
            project_name: str = "MORL-Baselines",
            experiment_name: str = "GPI-PD Continuous Action",
            wandb_entity: Optional[str] = None,
            log: bool = False,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            eval_iterations=1,
            self_evolution=True
    ):
        """GPI-PD algorithm with continuous actions.

        It extends the TD3 algorithm to multi-objective RL.
        It learns the policy and Q-networks conditioned on the weight vector.

        Args:
            env (gym.Env): The environment to train on.
            learning_rate (float, optional): The learning rate. Defaults to 3e-4.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft update coefficient. Defaults to 0.005.
            buffer_size (int, optional): The size of the replay buffer. Defaults to int(1e6).
            net_arch (List, optional): The network architecture for the policy and Q-networks.
            batch_size (int, optional): The batch size for training. Defaults to 256.
            num_q_nets (int, optional): The number of Q-networks to use. Defaults to 2.
            delay_policy_update (int, optional): The number of gradient steps to take before updating the policy. Defaults to 2.
            learning_starts (int, optional): The number of steps to take before starting to train. Defaults to 100.
            gradient_updates (int, optional): The number of gradient steps to take per update. Defaults to 1.
            use_gpi (bool, optional): Whether to use GPI for selecting actions. Defaults to True.
            policy_noise (float, optional): The noise to add to the policy. Defaults to 0.2.
            noise_clip (float, optional): The noise clipping value. Defaults to 0.5.
            per (bool, optional): Whether to use prioritized experience replay. Defaults to False.
            min_priority (float, optional): The minimum priority to use for prioritized experience replay. Defaults to 0.1.
            alpha (float, optional): The alpha value for prioritized experience replay. Defaults to 0.6.
            dyna (bool, optional): Whether to use Dyna. Defaults to False.
            project_name (str, optional): The name of the project. Defaults to "MORL Baselines".
            experiment_name (str, optional): The name of the experiment. Defaults to "GPI-PD Continuous Action".
            wandb_entity (Optional[str], optional): The wandb entity. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): The seed to use. Defaults to None.
            device (Union[th.device, str], optional): The device to use for training. Defaults to "auto".
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.use_gpi = use_gpi
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets
        self.delay_policy_update = delay_policy_update
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.per = per
        self.min_priority = min_priority
        self.alpha = alpha
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=buffer_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=buffer_size
            )

        self.q_nets = [
            QNetwork(self.observation_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        self.target_q_nets = [
            QNetwork(self.observation_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
            for param in target_q_net.parameters():
                param.requires_grad = False

        self.policy = Policy(
            self.observation_dim, self.reward_dim, self.action_dim, self.env.action_space, net_arch=net_arch
        ).to(self.device)
        self.target_policy = Policy(
            self.observation_dim, self.reward_dim, self.action_dim, self.env.action_space, net_arch=net_arch
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        for param in self.target_policy.parameters():
            param.requires_grad = False

        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)
        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=self.learning_rate)

        self.weight_support = []
        self.weight_support_ = [[0, 0]]
        self.stacked_weight_support = []
        self.EU_list = []
        self.visited_weights = []
        self.ccs = []
        self.demo_repository = []
        self.eval_iterations = eval_iterations
        self._n_updates = 0
        self.linear_support = LinearSupport(num_objectives=self.reward_dim,
                                            epsilon=0.0)
        self.log = log
        self.experiment_name = experiment_name
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)
        self.self_evolution = self_evolution

    def get_config(self):
        """Get the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "num_q_nets": self.num_q_nets,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "policy_noise": self.policy_noise,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "delay_policy_update": self.delay_policy_update,
            "min_priority": self.min_priority,
            "per": self.per,
            "buffer_size": self.buffer_size,
            "alpha": self.alpha,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def save(self, save_dir="weights/", filename=None, save_replay_buffer=True):
        """Save the agent's weights and replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
        }
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            saved_params["q_net_" + str(i) + "_state_dict"] = q_net.state_dict()
            saved_params["target_q_net_" + str(i) + "_state_dict"] = target_q_net.state_dict()
        saved_params["q_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        saved_params["M"] = self.weight_support
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the agent weights from a file."""
        params = th.load(path, map_location=self.device)
        self.weight_support = params["M"]
        self.stacked_weight_support = th.stack(self.weight_support)
        self.policy.load_state_dict(params["policy_state_dict"])
        self.policy_optim.load_state_dict(params["policy_optimizer_state_dict"])
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(params["q_net_" + str(i) + "_state_dict"])
            target_q_net.load_state_dict(params["target_q_net_" + str(i) + "_state_dict"])
        self.q_optim.load_state_dict(params["q_nets_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self, weight: th.Tensor):
        """Update the policy and the Q-nets."""
        for _ in range(self.gradient_updates):
            if self.per:
                (s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes) = self._sample_batch_experiences()
            else:
                (s_obs, s_actions, s_rewards, s_next_obs, s_dones) = self._sample_batch_experiences()

            if len(self.weight_support) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = (
                    s_obs.repeat(2, 1),
                    s_actions.repeat(2, 1),
                    s_rewards.repeat(2, 1),
                    s_next_obs.repeat(2, 1),
                    s_dones.repeat(2, 1),
                )
                w = th.vstack(
                    [weight for _ in range(s_obs.size(0) // 2)] + random.choices(self.weight_support,
                                                                                 k=s_obs.size(0) // 2)
                )
            else:
                w = weight.repeat(s_obs.size(0), 1)

            with th.no_grad():
                next_actions = self.target_policy(s_next_obs, w, noise=self.policy_noise, noise_clip=self.noise_clip)
                q_targets = th.stack([q_target(s_next_obs, next_actions, w) for q_target in self.target_q_nets])
                scalarized_q_targets = th.einsum("nbr,br->nb", q_targets, w)
                inds = th.argmin(scalarized_q_targets, dim=0, keepdim=True)
                inds = inds.reshape(1, -1, 1).expand(1, q_targets.size(1), q_targets.size(2))
                target_q = q_targets.gather(0, inds).squeeze(0)

                target_q = (s_rewards + (1 - s_dones) * self.gamma * target_q).detach()

            q_values = [q_net(s_obs, s_actions, w) for q_net in self.q_nets]
            critic_loss = (1 / self.num_q_nets) * sum([F.mse_loss(q_value, target_q) for q_value in q_values])

            self.q_optim.zero_grad()
            critic_loss.backward()
            self.q_optim.step()

            if self.per:
                per = (q_values[0] - target_q)[: len(idxes)].detach().abs() * 0.05
                per = th.einsum("br,br->b", per, w[: len(idxes)])
                priority = per.cpu().numpy().flatten()
                priority = priority.clip(min=self.min_priority) ** self.alpha
                self.replay_buffer.update_priorities(idxes, priority)

            for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
                polyak_update(q_net.parameters(), target_q_net.parameters(), self.tau)

            if self._n_updates % self.delay_policy_update == 0:
                # Policy update
                actions = self.policy(s_obs, w)
                q_values_pi = (1 / self.num_q_nets) * sum(q_net(s_obs, actions, w) for q_net in self.q_nets)
                policy_loss = -th.einsum("br,br->b", q_values_pi, w).mean()

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                polyak_update(self.policy.parameters(), self.target_policy.parameters(), self.tau)

            self._n_updates += 1

        if self.log and self.global_step % 100 == 0:
            if self.per:
                wandb.log(
                    {
                        "metrics/mean_priority": np.mean(priority),
                        "metrics/max_priority": np.max(priority),
                        "metrics/min_priority": np.min(priority),
                    },
                    commit=False,
                )
            wandb.log(
                {
                    "losses/critic_loss": critic_loss.item(),
                    "losses/policy_loss": policy_loss.item(),
                    "global_step": self.global_step,
                },
            )

    @th.no_grad()
    def eval(
            self, obs: Union[np.ndarray, th.Tensor], w: Union[np.ndarray, th.Tensor], torch_action=False
    ) -> Union[np.ndarray, th.Tensor]:
        """Evaluate the policy action for the given observation and weight vector."""
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs).float().to(self.device)
            w = th.tensor(w).float().to(self.device)

        if self.use_gpi:
            obs = obs.repeat(len(self.weight_support), 1)
            actions_original = self.policy(obs, self.stacked_weight_support)

            obs = obs.repeat(len(self.weight_support), 1, 1)
            actions = actions_original.repeat(len(self.weight_support), 1, 1)
            stackedM = self.stacked_weight_support.repeat_interleave(len(self.weight_support), dim=0).view(
                len(self.weight_support), len(self.weight_support), self.reward_dim
            )
            values = self.q_nets[0](obs, actions, stackedM)

            scalar_values = th.einsum("par,r->pa", values, w)
            max_q, a = th.max(scalar_values, dim=1)
            policy_index = th.argmax(max_q)  # max_i max_a q(s,a,w_i)
            action = a[policy_index].detach().item()
            action = actions_original[action]
        else:
            action = self.policy(obs, w)

        if not torch_action:
            action = action.detach().cpu().numpy()

        return action

    def set_weight_support(self, weight_list: List[np.ndarray]):
        """Set the weight support set."""
        weights_no_repeat = unique_tol(weight_list)
        self.weight_support = [th.tensor(w).float().to(self.device) for w in weights_no_repeat]
        if len(self.weight_support) > 0:
            self.stacked_weight_support = th.stack(self.weight_support)

    def scale_action(self, action, low, high):
        action = np.clip(action, -1, 1)
        weight = (high - low) / 2
        bias = (high + low) / 2
        action_ = action * weight + bias

        return action_

    def add_new_demo_info(self, new_demo, disc_vec_return, horizon):
        if horizon is None:
            d_info = demo_info(demo=new_demo, disc_vec_return=disc_vec_return, updated=False, horizon=len(new_demo))
        else:
            d_info = demo_info(demo=new_demo, disc_vec_return=disc_vec_return, updated=False, horizon=horizon)
        self.demo_repository.append(d_info)
        return d_info

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
                    f"demo:{i} - vec:{np.round_(self.demo_repository[i].disc_vec_return, 4)} is removed")
                self.demo_repository.pop(i)

    def get_corners_(self):
        corners = self.linear_support.compute_corner_weights()
        for w in corners:
            self.linear_support.visited_weights.append(w)
        return corners

    def get_demos_EU_(self, eval_weights, eval_env):
        utility_thresholds = []
        for w_c in eval_weights:
            demo, utility_threshold, _, _ = self.weight_to_demo_(w=w_c, eval_env=eval_env,
                                                                 eval_iterations=self.eval_iterations)
            utility_thresholds.append(utility_threshold)
        utility_thresholds = np.array(utility_thresholds)
        EU_target = np.mean(utility_thresholds)
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
            # if verbose:
            #     print(f"w:{weight}\tdisc_vec_return:{disc_vec_return}")
            utilities.append(disc_return)
            if show_case:
                print(f"for w:{np.round_(weight, 3)}\tdisc vec return:{disc_vec_return}")
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
                                                                                      eval_iterations=eval_iterations)
            disc_scalar_returns.append(utility)
            disc_vec_returns.append(disc_vec_return)
        max_demo_idx = np.argmax(disc_scalar_returns)
        max_demo_info = self.demo_repository[max_demo_idx]
        max_utility = disc_scalar_returns[max_demo_idx]
        max_vec_return = disc_vec_returns[max_demo_idx]
        return max_demo_info, max_utility, max_vec_return, max_demo_idx

    def play_a_episode_(self, env, weight, demo_info: demo_info, evaluation=False, max_steps=50, eval_iterations=5):

        terminated = False
        truncated = False

        new_demo = []
        action_pointer = 0
        steps = 0

        obs, _ = env.reset()
        while not terminated and not truncated:
            steps += 1
            if action_pointer < demo_info.horizon:
                action = demo_info.demo[action_pointer]
                action_pointer += 1
            else:
                with th.no_grad():
                    action = (agent.policy(th.tensor(obs).float().to(self.device),
                                           th.as_tensor(weight).float().to(self.device),
                                           noise=self.policy_noise,
                                           noise_clip=self.noise_clip, ).detach().cpu().numpy())

            action_ = self.scale_action(action, low=eval_env.action_space.low, high=eval_env.action_space.high)

            obs_, rewards, terminated, truncated, _ = env.step(action_)
            obs = obs_
            new_demo.append(action)
        if evaluation:
            print(f"eval action traj:{new_demo}")

        avg_utility, avg_disc_vec_return, avg_scalar_return, avg_vec_return = self.evaluate_demo_(new_demo,
                                                                                                  eval_env,
                                                                                                  weights=weight,
                                                                                                  eval_iterations=eval_iterations)
        return avg_utility, avg_disc_vec_return, avg_scalar_return, avg_vec_return, new_demo

    def unique_tol(self, a: List[np.ndarray], tol=1e-4) -> List[np.ndarray]:
        """Returns unique elements of a list of np.arrays, within a tolerance."""
        if len(a) == 0:
            return a
        delete = np.array([False] * len(a))
        a = np.array(a)
        for i in range(len(a)):
            if delete[i]:
                continue
            for j in range(i + 1, len(a)):
                if np.allclose(a[i], a[j], tol):
                    delete[j] = True
        return list(a[~delete])

    def evaluate_demo_(self, demo, eval_env, weights=np.zeros([1, 0]), eval_iterations=5):
        # print(f"evaluate demo")
        disc_vec_return = np.zeros(self.reward_dim, dtype=np.float64)
        vec_return = np.zeros(self.reward_dim, dtype=np.float64)
        disc_scalar_return = 0
        scalar_return = 0
        for _ in range(eval_iterations):
            gamma = 1
            obs, _ = eval_env.reset()
            for action in demo:
                action_ = self.scale_action(action, low=eval_env.action_space.low, high=eval_env.action_space.high)
                _, rewards, terminated, truncated, _ = eval_env.step(action_)

                # print(f"rewards:{rewards}")
                disc_scalar_return += gamma * np.dot(rewards, weights)
                disc_vec_return += gamma * rewards

                scalar_return += np.dot(rewards, weights)
                vec_return += rewards

                gamma *= self.gamma
                if terminated or truncated:
                    break

        avg_disc_scalar_return = disc_scalar_return / eval_iterations
        avg_disc_vec_return = disc_vec_return / eval_iterations
        avg_scalar_return = scalar_return / eval_iterations
        avg_vec_return = vec_return / eval_iterations

        return avg_disc_scalar_return, avg_disc_vec_return, avg_scalar_return, avg_vec_return

    def select_w_(self, eval_env, weights, mode="get_max"):
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
            priority = abs((utility_threshold - u) / utility_threshold)
            priorities.append(priority)
            demos_info_.append(demo_info)
            u_thresholds.append(utility_threshold)
            max_disc_vec_returns.append(max_disc_vec_return)
        idx = random.choices(range(len(priorities)), weights=priorities, k=1)[0]
        # idx = np.argmax(priorities)
        w = weights[idx]
        demo_info = demos_info_[idx]
        utility_threshold = u_thresholds[idx]
        max_disc_vec_return = max_disc_vec_returns[idx]
        print(f"select:{w}by priority of:{priorities[idx]} @ mode:{mode}\tdemo_horizon:{demo_info.horizon}")
        return w, demo_info, utility_threshold, max_disc_vec_return

    def jsmorl_train(self,
                     demos,
                     eval_env,
                     total_timesteps,
                     timesteps_per_iter,
                     title="try",
                     eval_freq=100,
                     roll_back_step=5,
                     initial_pass_thres=0.8
                     ):
        for demo in demos:
            avg_utility, avg_disc_vec_return, avg_scalar_return, avg_vec_return = self.evaluate_demo_(demo,
                                                                                                      eval_env,
                                                                                                      weights=np.zeros(
                                                                                                          self.reward_dim))
            print(f"demo vec:{avg_disc_vec_return}")
            self.add_new_demo_info(new_demo=demo, disc_vec_return=avg_disc_vec_return, horizon=None)

        self.update_linear_support_ccs()
        corners = self.get_corners_()
        print(f"corner weights:{np.round_(corners, 3)}")
        self.weight_support = []
        self.set_weight_support(corners)
        eval_weights = equally_spaced_weights(self.reward_dim, n=100)
        EU_target = self.get_demos_EU_(eval_weights=eval_weights, eval_env=eval_env)
        print(f"@{self.global_step}\tEU_target:{EU_target}")
        while self.global_step < total_timesteps:

            self.jsmorl_train_iteration(eval_env=eval_env,
                                        eval_freq=eval_freq,
                                        weight_support=corners,
                                        total_timesteps=timesteps_per_iter,
                                        roll_back_step=roll_back_step,
                                        path=title,
                                        initial_pass_thres=initial_pass_thres)

            self.update_linear_support_ccs()
            self.remove_obsolete_demos()
            for i in range(len(self.demo_repository)):
                demo_info = self.demo_repository[i]
                # print(f"@REPO --> demo len:{len(demo_info.demo)}\tvec_return {np.round_(demo_info.disc_vec_return, 4)}")

            self.renew_horizon()
            corners = self.get_corners_()
            self.weight_support = []
            self.set_weight_support(corners)
        print("----train_finish----")

        # self.get_agent_EU_(eval_env=eval_env, eval_weights=eval_weights, eval_iterations=1, show_case=True, path=title)
        file_name = title + "/_15.npy"
        np.save(file_name, self.EU_list)
        print("--saved--")
        # corners = self.get_corners_()
        # print(f"corner weights:{np.round_(corners, 3)}")
        # eval_weights = equally_spaced_weights(self.reward_dim, n=100)
        # self.get_agent_EU_(eval_env=eval_env,
        #                    eval_weights=eval_weights,
        #                    eval_iterations=self.eval_iterations,
        #                    show_case=True)
        # plt.plot(self.EU_list)
        # plt.show()

    def jsmorl_train_iteration(self,
                               eval_env=None,
                               eval_freq: int = 100,
                               weight_support=None,
                               total_timesteps=4000,
                               roll_back_step=100,
                               path=None,
                               initial_pass_thres=0.8):

        eval_weights = equally_spaced_weights(self.reward_dim, n=100)

        candidate_ws = random.choices(eval_weights, k=len(weight_support))
        weight_support += candidate_ws
        w, demo_info, utility_threshold, disc_vec_return = self.select_w_(eval_env=eval_env,
                                                                          weights=weight_support)

        self.linear_support.visited_weights.append(w)
        tensor_w = th.tensor(w).float().to(self.device)
        # EU_target = self.get_demos_EU_(eval_weights=eval_weights, eval_env=eval_env)
        # print(f"EU_target:{EU_target}")
        pi_g_pointer = 0
        obs, _ = self.env.reset()
        step = 0
        departed = False
        #
        # EU_target = self.get_demos_EU_(eval_weights=eval_weights, eval_env=eval_env)
        # print(f"@{self.global_step}\tEU_target:{EU_target}")
        u_thres_factor = min(initial_pass_thres + 0.05 * (self.global_step // 1.5e4), 0.99)
        while step < total_timesteps and demo_info.horizon >= 0:
            step += 1
            self.global_step += 1
            if departed == False:
                if demo_info.horizon > 0 and pi_g_pointer < demo_info.horizon and u_thres_factor < 1:
                    action = demo_info.demo[:demo_info.horizon][pi_g_pointer]
                    pi_g_pointer += 1
                else:
                    with th.no_grad():
                        action = (agent.policy(th.tensor(obs).float().to(self.device),
                                               th.as_tensor(w).float().to(self.device),
                                               noise=self.policy_noise,
                                               noise_clip=self.noise_clip, ).detach().cpu().numpy())
            else:
                with th.no_grad():
                    action = (agent.policy(th.tensor(obs).float().to(self.device),
                                           th.as_tensor(w).float().to(self.device),
                                           noise=self.policy_noise,
                                           noise_clip=self.noise_clip, ).detach().cpu().numpy())

            action_ = self.scale_action(action, low=eval_env.action_space.low, high=eval_env.action_space.high)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action_)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                if self.global_step == self.learning_starts:
                    print(">>>>>>>>>>>>>START TRAIN<<<<<<<<<<<<<<<")
                self.update(tensor_w)

            if self.global_step % eval_freq == 0:
                u, disc_vec_return, _, _, new_demo = self.play_a_episode_(env=eval_env,
                                                                          weight=w,
                                                                          demo_info=demo_info,
                                                                          evaluation=False
                                                                          )
                u_thres_factor = min(initial_pass_thres + 0.05 * (self.global_step // 1.5e4), 0.99)
                # u_thres_factor = 1
                # print(f"@{self.global_step}--"
                #       f"weight:{w}\t"
                #       f"guide policy horizon:{demo_info.horizon}\t"
                #       f"new demo len:{len(new_demo)}\t"
                #       f"u:{np.round_(u, 4)}\t"
                #       f"u_threshold:{np.round_(utility_threshold, 4)}")

                if np.round_(u, 4) >= np.round_(utility_threshold, 4) * u_thres_factor:
                    # print(f"-- reach threshold --"
                    #       f"u:{np.round_(u, 4)}>u_threshold:{np.round_(utility_threshold, 4)}\t"
                    #       f"guide policy horizon:{demo_info.horizon}\tu_thres_factor:{u_thres_factor}")

                    if demo_info.horizon >= 1:
                        demo_info.horizon = max(demo_info.horizon - roll_back_step, 0)
                    else:
                        demo_info.horizon -= roll_back_step
                    if self.self_evolution:
                        if np.round_(u, 4) > np.round_(utility_threshold, 4):
                            demo_info.updated = True
                            print(f"replace old demo {len(demo_info.demo)} with better demo:{len(new_demo)}")
                            demo_info = self.add_new_demo_info(new_demo=new_demo,
                                                               disc_vec_return=disc_vec_return,
                                                               horizon=demo_info.horizon)

            if self.global_step % total_timesteps == 0:
                EU, new_demos = self.get_agent_EU_(eval_weights=eval_weights, eval_env=eval_env,
                                                   eval_iterations=self.eval_iterations, show_case=True, path=path,
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
                      )

                self.EU_list.append(EU)

            if terminated or truncated:
                if not departed:
                    departed = True
                if departed:
                    departed = False

                self.police_indices = []
                pi_g_pointer = 0
                # w, demo_info, utility_threshold, disc_vec_return = self.select_w_(eval_env=eval_env,
                #                                                                   weights=weight_support)
                # tensor_w = th.tensor(w).float().to(self.device)

                obs, _ = self.env.reset()
            else:
                obs = next_obs

        return self.demo_repository


class GPILSContinuousAction(GPIPDContinuousAction):
    """Model-free version of GPI-PD with continuous actions."""

    def __init__(self, *args, **kwargs):
        """Initialize the agent deactivating the dynamics model."""
        super().__init__(dyna=False, experiment_name="GPI-LS Continuous Action", *args, **kwargs)


if __name__ == '__main__':
    env_names = [
        # "mo_hopper",
                 # "mo_ant",
                 "mo_humanoid"
                 ]
    seeds = [
        2,
        7,
        15,
        42,
        78
    ]

    print(f"seeds:{seeds}")
    experiment_type = "ablation"
    roll_backs = [100]
    # experiment_title = "Threshold"
    for env_name in env_names:
        for seed in seeds:
            # env_name = "mo_humanoid"
            print(f"env_name:{env_name}, seed:{seed} now running")
            save_dir = "../results/" + env_name + "/" + experiment_type + "_seed_" + str(seed)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            start_time = time.time()
            if env_name == "mo_hopper":
                # ------------------ make MO Hopper --------------------#
                env = mo_gym.make("mo-hopper-v4", cost_objective=False, max_episode_steps=500, reset_noise_scale=0)
                eval_env = mo_gym.make("mo-hopper-v4", cost_objective=False, max_episode_steps=500, reset_noise_scale=0)
                human_demos_1 = np.load("..//guide_demos//mo_hopper//mo_hopper_demos[1. 0.].npy", allow_pickle=True)
                human_demos_2 = np.load("..//guide_demos//mo_hopper//mo_hopper_demos[0.75 0.25].npy", allow_pickle=True)
                human_demos_3 = np.load("..//guide_demos//mo_hopper//mo_hopper_demos[0.5 0.5].npy", allow_pickle=True)
                human_demos_4 = np.load("..//guide_demos//mo_hopper//mo_hopper_demos[0.25 0.75].npy", allow_pickle=True)
                human_demos_5 = np.load("..//guide_demos//mo_hopper//mo_hopper_demos[0. 1.].npy", allow_pickle=True)
                threshold = 0.8
                demo_idx = -25

            if env_name == "mo_ant":
                # ------------------- make multi-objective ant --------------------#
                env = mo_gym.make("mo-ant-v4", cost_objective=False, max_episode_steps=500, reset_noise_scale=0)
                eval_env = mo_gym.make("mo-ant-v4", cost_objective=False, max_episode_steps=500, reset_noise_scale=0)
                human_demos_1 = np.load("TD3//MO_ant_demos[1. 0.].npy", allow_pickle=True)
                human_demos_2 = np.load("TD3//MO_ant_demos[0.75 0.25].npy", allow_pickle=True)
                human_demos_3 = np.load("TD3//MO_ant_demos[0.5 0.5].npy", allow_pickle=True)
                human_demos_4 = np.load("TD3//MO_ant_demos[0.25 0.75].npy", allow_pickle=True)
                human_demos_5 = np.load("TD3//MO_ant_demos[0. 1.].npy", allow_pickle=True)
                threshold = 0.6
                demo_idx = -25

            if env_name == "mo_humanoid":
                # ------------------ make humanoid --------------------#
                env = mo_gym.make("mo-humanoid-v4", max_episode_steps=500, reset_noise_scale=0)
                eval_env = mo_gym.make("mo-humanoid-v4", max_episode_steps=500, reset_noise_scale=0)
                human_demos_1 = np.load("TD3//MO_humanoid_demos[1. 0.].npy", allow_pickle=True)
                human_demos_2 = np.load("TD3//MO_humanoid_demos[0.75 0.25].npy", allow_pickle=True)
                human_demos_3 = np.load("TD3//MO_humanoid_demos[0.5 0.5].npy", allow_pickle=True)
                human_demos_4 = np.load("TD3//MO_humanoid_demos[0.25 0.75].npy", allow_pickle=True)
                human_demos_5 = np.load("TD3//MO_humanoid_demos[0. 1.].npy", allow_pickle=True)
                threshold = 0.6
                demo_idx = -1

            agent = GPIPDContinuousAction(
                env,
                # gradient_updates=g,
                min_priority=0.1,
                batch_size=128,
                buffer_size=int(4e5),
                per=True,
                project_name="MORL-Baselines",
                experiment_name=env_name + "_" + str(seed),
                log=False,
                eval_iterations=1,
                seed=seed,
                gradient_updates=20,
                self_evolution=False
            )

            print(len(human_demos_1[-1]))
            # demo_idx = -1
            # demo_idx = -25
            human_demos = [human_demos_1[demo_idx],
                           human_demos_2[demo_idx],
                           human_demos_3[demo_idx],
                           human_demos_4[demo_idx],
                           human_demos_5[demo_idx]
                           ]
            # human_demos = random.choices(human_demos, k=3)

            agent.jsmorl_train(demos=human_demos, eval_env=eval_env, total_timesteps=15 * 1.5e4,
                               timesteps_per_iter=1.5e4,
                               title=save_dir,
                               eval_freq=100,
                               roll_back_step=100,
                               initial_pass_thres=threshold)
            end_time = time.time()

            elapsed_time_in_seconds = end_time - start_time
            elapsed_time_in_hours = elapsed_time_in_seconds / 3600
            print(f"Run {experiment_type} with seed {seed} for {elapsed_time_in_hours:.2f} hours")
