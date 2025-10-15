"""Linear Support implementation."""
import random
from copy import deepcopy
from typing import List, Optional, Tuple

import cdd
import cvxpy as cp
import numpy as np
from cvxpy import SolverError
from gymnasium.core import Env
from Algorithm.common.weights import equally_spaced_weights
from Algorithm.common.evaluation import eval_mo_demo
import mo_gymnasium as mo_gym


def eval_mo(
        agent,
        env,
        w: Optional[np.ndarray] = None,
        scalarization=np.dot,
        render: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment.

    Args:
        agent: Agent
        env: MO-Gymnasium environment with LinearReward wrapper
        scalarization: scalarization function, taking weights and reward as parameters
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return
    """
    obs, _ = env.reset()
    done = False
    vec_return, disc_vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    while not done:
        if render:
            env.render()
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, w))
        done = terminated or truncated
        vec_return += r
        disc_vec_return += gamma * r
        gamma *= agent.gamma

    if w is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    )


def policy_evaluation_mo(agent, env, w: np.ndarray, rep: int = 5) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        w (np.ndarray): Weight vector
        rep (int, optional): Number of episodes for averaging. Defaults to 5.

    Returns:
        (float, float, np.ndarray, np.ndarray): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return
    """
    evals = [eval_mo(agent, env, w) for _ in range(rep)]
    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)

    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
    )


def random_weights(
        dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w


def extrema_weights(dim: int) -> List[np.ndarray]:
    """Generate weight vectors in the extrema of the weight simplex. That is, one element is 1 and the rest are 0.

    Args:
        dim: size of the weight vector
    """
    return list(np.eye(dim, dtype=np.float32))


class LinearSupport:
    """Linear Support for computing corner weights when using linear utility functions.

    Implements both

    Optimistic Linear Support (OLS) algorithm:
    Paper: (Section 3.3 of http://roijers.info/pub/thesis.pdf).

    Generalized Policy Improvement Linear Support (GPI-LS) algorithm:
    Paper: https://arxiv.org/abs/2301.07784
    """

    def __init__(
            self,
            num_objectives: int,
            epsilon: float = 0.0,
            verbose: bool = True,
    ):
        """Initialize Linear Support.

        Args:
            num_objectives (int): Number of objectives
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to False.
        """
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.visited_weights = []  # List of already tested weight vectors
        self.ccs = []
        self.weight_support = []  # List of weight vectors for each value vector in the CCS
        self.queue = []
        self.iteration = 0
        self.verbose = verbose
        self.policies = []
        # self.demo_support_weights = []
        for w in extrema_weights(self.num_objectives):
            self.queue.append((float("inf"), w))

    def get_support_weight_from_demo(self, demos, env):
        for i in range(len(demos)):
            _, discounted_return, _, disc_vec_return = eval_mo_demo(demo=demos[i],
                                                                    env=env,
                                                                    w=np.array([1, 0], dtype=float))
            self.ccs.append(disc_vec_return)
        corners = self.compute_corner_weights()

        def find_w_demo(w):
            max_u = -np.inf
            max_demo = demos[0]
            for i in range(len(demos)):
                _, _, _, discounted_vec_return = eval_mo_demo(demo=demos[i],
                                                              env=env,
                                                              w=np.array([1, 0], dtype=float))
                u = np.dot(w, discounted_vec_return)
                if u > max_u:
                    max_u = u
                    max_demo = demos[i]
            max_u = int(max_u*100)/100.0
            return max_demo, max_u

        demo_list = []
        u_thresholds = []
        for w in corners:
            self.weight_support.append(w)
            self.visited_weights.append(w)
            demo, utility = find_w_demo(w)
            demo_list.append(demo)
            u_thresholds.append(utility)
        return corners, demo_list, u_thresholds

    def next_weight(self, algo: str = "ols", gpi_agent=None, env=None, rep_eval=1
                    ):
        """Returns the next weight vector with highest priority.
        Args:
            algo (str): Algorithm to use. Either 'ols' or 'gpi-ls'.
        Returns:
            np.ndarray: Next weight vector
        """
        if len(self.ccs) > 0:
            W_corner = self.compute_corner_weights()
            if self.verbose:
                print("W_corner:", W_corner, "W_corner size:", len(W_corner))

            self.queue = []
            for wc in W_corner:
                if algo == "ols":
                    priority = self.ols_priority(wc)
                # print(f"wc:{wc}\tpriority:{priority}")
                elif algo == "gpi-ls":
                    if gpi_agent is None:
                        raise ValueError("GPI-LS requires passing a GPI agent.")
                    gpi_expanded_set = [policy_evaluation_mo(gpi_agent, env, wc, rep=rep_eval)[3] for wc in W_corner]
                    priority = self.gpi_ls_priority(wc, gpi_expanded_set)

                if self.epsilon is None or priority >= self.epsilon:
                    # OLS does not try the same weight vector twice
                    if not (algo == "ols" and any([np.allclose(wc, wv) for wv in self.visited_weights])):
                        self.queue.append((priority, wc))

            if len(self.queue) > 0:
                # Sort in descending order of priority
                self.queue.sort(key=lambda t: t[0], reverse=True)
                # If all priorities are 0, shuffle the queue to avoid repeating weights every iteration
                if self.queue[0][0] == 0.0:
                    random.shuffle(self.queue)

        if self.verbose:
            print("CCS:", self.ccs, "CCS size:", len(self.ccs))

        if len(self.queue) == 0:
            if self.verbose:
                print("There are no corner weights in the queue. Returning None.")
            return None
        else:
            next_w = self.queue.pop(0)[1]
            if self.verbose:
                print("Next weight:", next_w)
            return next_w

    def get_weight_support(self) -> List[np.ndarray]:
        """Returns the weight support of the CCS.

        Returns:
            List[np.ndarray]: List of weight vectors of the CCS

        """
        return deepcopy(self.weight_support)

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        """Returns the corner weights of the current CCS.

        Args:
            top_k: If not None, returns the top_k corner weights.

        Returns:
            List[np.ndarray]: List of corner weights.
        """
        weights = [w.copy() for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    def ended(self) -> bool:
        """Returns True if the queue is empty."""
        return len(self.queue) == 0

    def add_solution(self, value: np.ndarray, w: np.ndarray) -> List[int]:
        """Add new value vector optimal to weight w.

        Args:
            value (np.ndarray): New value vector
            w (np.ndarray): Weight vector

        Returns:
            List of indices of value vectors removed from the CCS for being dominated.
        """
        if self.verbose:
            print(f"Adding value: {value} to CCS.")

        self.iteration += 1
        self.visited_weights.append(w)

        if self.is_dominated(value):
            if self.verbose:
                print(f"Value {value} is dominated. Discarding.")
            return [len(self.ccs)]

        removed_indx = self.remove_obsolete_values(value)

        self.ccs.append(value)
        self.weight_support.append(w)

        return removed_indx

    def ols_priority(self, w: np.ndarray) -> float:
        """Get the priority of a weight vector for OLS.

        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """
        max_value_ccs = self.max_scalarized_value(w)
        max_optimistic_value = self.max_value_lp(w)
        priority = max_optimistic_value - max_value_ccs
        return priority

    def gpi_ls_priority(self, w: np.ndarray, gpi_expanded_set: List[np.ndarray]) -> float:
        """Get the priority of a weight vector for GPI-LS.
        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """

        def best_vector(values, w):
            max_v = values[0]
            for i in range(1, len(values)):
                if values[i] @ w > max_v @ w:
                    max_v = values[i]
            return max_v

        max_value_ccs = self.max_scalarized_value(w)
        max_value_gpi = best_vector(gpi_expanded_set, w)
        max_value_gpi = np.dot(max_value_gpi, w)
        priority = max_value_gpi - max_value_ccs

        return priority

    def max_scalarized_value(self, w: np.ndarray) -> Optional[float]:
        """Returns the maximum scalarized value for weight vector w.

        Args:
            w: Weight vector

        Returns:
            Maximum scalarized value for weight vector w.
        """
        if len(self.ccs) == 0:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def remove_obsolete_weights(self, new_value: np.ndarray) -> List[np.ndarray]:
        """Remove from the queue the weight vectors for which the new value vector is better than previous values.

        Args:
            new_value: New value vector

        Returns:
            List of weight vectors removed from the queue.
        """
        if len(self.ccs) == 0:
            return []
        W_del = []
        inds_remove = []
        for i, (priority, cw) in enumerate(self.queue):
            if np.dot(cw, new_value) > self.max_scalarized_value(cw):
                W_del.append(cw)
                inds_remove.append(i)
        for i in reversed(inds_remove):
            self.queue.pop(i)
        return W_del

    def remove_obsolete_values(self, value: np.ndarray) -> List[int]:
        """Removes the values vectors which are no longer optimal for any weight vector after adding the new value vector.

        Args:
            value (np.ndarray): New value vector

        Returns:
            The indices of the removed values.
        """
        removed_indx = []
        for i in reversed(range(len(self.ccs))):
            weights_optimal = []
            for w in self.visited_weights:
                # find the corresponding CCS point to w
                if np.dot(self.ccs[i], w) == self.max_scalarized_value(w):
                    # if this point is better than the input value based on this w,
                    # it means this the optimal value for this w is found
                    if np.dot(value, w) <= np.dot(self.ccs[i], w):
                        weights_optimal.append(w)
            if len(weights_optimal) == 0:
                print("removed value", self.ccs[i])
                removed_indx.append(i)
                self.ccs.pop(i)
                self.weight_support.pop(i)
        return removed_indx

    def JS_add_solution(self, value, w, demo):
        demos = []
        if self.verbose:
            print(f"Adding value: {value} to CCS.")

        if self.is_dominated(value):
            if self.verbose:
                print(f"Value {value} is dominated. Discarding.")
            return [len(self.ccs)]

        removed_indx = self.remove_obsolete_values(value)

        # self.ccs.append(value)
        self.weight_support.append(w)
        demos.append(demo)

        return removed_indx

    def JS_remove_obsolete_values(self):
        pass

    def max_value_lp(self, w_new: np.ndarray) -> float:
        """Returns an upper-bound for the maximum value of the scalarized objective.

        Args:
            w_new: New weight vector

        Returns:
            Upper-bound for the maximum value of the scalarized objective.
        """
        # No upper bound if no values in CCS
        if len(self.ccs) == 0:
            return float("inf")

        w = cp.Parameter(self.num_objectives)
        w.value = w_new
        v = cp.Variable(self.num_objectives)

        W_ = np.vstack(self.visited_weights)
        W = cp.Parameter(W_.shape)
        W.value = W_

        V_ = np.array([self.max_scalarized_value(weight) for weight in self.visited_weights])
        V = cp.Parameter(V_.shape)
        # print(f"V_:{V_}")
        V.value = V_

        # Maximum value for weight vector w
        objective = cp.Maximize(w @ v)
        # such that it is consistent with other optimal values for other visited weights
        constraints = [W @ v <= V]
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(verbose=False)
        except SolverError:
            print("ECOS solver error, trying another one.")
            result = prob.solve(solver=cp.SCS, verbose=False)
        return result

    def compute_corner_weights(self) -> List[np.ndarray]:
        """Returns the corner weights for the current set of values.

        See http://roijers.info/pub/thesis.pdf Definition 19.
        Obs: there is a typo in the definition of the corner weights in the thesis, the >= sign should be <=.

        Returns:
            List of corner weights.
        """
        A = np.vstack(self.ccs)
        A = np.round_(A, decimals=4)  # Round to avoid numerical issues
        A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1, 1)), axis=1)

        A_plus = np.ones(A.shape[1]).reshape(1, -1)
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)
        A_plus = -np.ones(A.shape[1]).reshape(1, -1)
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)

        for i in range(self.num_objectives):
            A_plus = np.zeros(A.shape[1]).reshape(1, -1)
            A_plus[0, i] = -1
            A = np.concatenate((A, A_plus), axis=0)

        b = np.zeros(len(self.ccs) + 2 + self.num_objectives)
        b[len(self.ccs)] = 1
        b[len(self.ccs) + 1] = -1

        def compute_poly_vertices(A, b):
            # Based on https://stackoverflow.com/questions/65343771/solve-linear-inequalities
            b = b.reshape((b.shape[0], 1))
            mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
            mat.rep_type = cdd.RepType.INEQUALITY
            P = cdd.Polyhedron(mat)
            g = P.get_generators()
            V = np.array(g)
            vertices = []
            for i in range(V.shape[0]):
                if V[i, 0] != 1:
                    continue
                if i not in g.lin_set:
                    vertices.append(V[i, 1:])
            return vertices

        vertices = compute_poly_vertices(A, b)
        corners = []
        for v in vertices:
            corners.append(v[:-1])

        return corners

    def is_dominated(self, value: np.ndarray) -> bool:
        """Checks if the value is dominated by any of the values in the CCS.

        Args:
            value: Value vector

        Returns:
            True if the value is dominated by any of the values in the CCS, False otherwise.
        """
        if len(self.ccs) == 0:
            return False
        for w in self.visited_weights:
            if np.dot(value, w) >= self.max_scalarized_value(w):
                return False
        return True

    def train(
            self,
            total_timesteps: int,
            timesteps_per_iteration: int = int(2e5),
    ):
        """Learn a set of policies.

        Args:
            total_timesteps: The total number of timesteps to train for.
            timesteps_per_iteration: The number of timesteps per iteration.
        """
        num_iterations = int(total_timesteps / timesteps_per_iteration)

        for _ in range(num_iterations):
            w = self.next_weight()
            if w is None:
                print("OLS has no more corner weights to try. Using a random weight instead.")
                w = random_weights(2)

            value = _solve(w)
            self.add_solution(value, w)


if __name__ == "__main__":
    action_demo_1 = [1]  # 0.7
    action_demo_2 = [3, 1, 1]  # 8.2
    action_demo_3 = [3, 3, 1, 1, 1]  # 11.5
    action_demo_4 = [3, 3, 3, 1, 1, 1, 1]  # 14.0
    action_demo_5 = [3, 3, 3, 3, 1, 1, 1, 1]  # good.1
    action_demo_6 = [3, 3, 3, 3, 3, 1, 1, 1, 1]  # 16.1
    action_demo_7 = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 19.6
    action_demo_8 = [3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 20.3
    action_demo_9 = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 22.4
    action_demo_10 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 23.7


    def _solve(w):
        if w[1] > 0.70:
            return np.array([-17.38, 19.78])
        elif w[1] > 0.67:
            return np.array([-15.71, 19.07])
        elif w[1] > 0.66:
            return np.array([-13.13, 17.81])
        elif w[1] > 0.58:
            return np.array([-12.25, 17.37])
        elif w[1] > 0.54:
            return np.array([-8.65, 14.86])
        elif w[1] > 0.51:
            return np.array([-7.73, 14.07])
        elif w[1] > 0.47:
            return np.array([-6.79, 13.18])
        elif w[1] > 0.39:
            return np.array([-4.9, 11.05])
        elif w[1] > 0.21:
            return np.array([-2.97, 8.04])
        else:
            return np.array([-1., 0.7])
        # return np.array(list(map(float, input().split())), dtype=np.float32)


    action_demos = [action_demo_1, action_demo_2, action_demo_3, action_demo_4, action_demo_5, action_demo_6,
                    action_demo_7, action_demo_8, action_demo_9, action_demo_10]
    eval_env = mo_gym.make("deep-sea-treasure-v0")
    num_objectives = 2
    ols = LinearSupport(num_objectives=num_objectives, epsilon=0.0001, verbose=True)
    corners, demos, u_threshold = ols.get_support_weight_from_demo(demos=action_demos, env=eval_env)
    print(corners)
    for w_d in zip(corners, demos, u_threshold):
        print(f"corner_w:{w_d[0]}\tU:{w_d[2]}\tdemo:{w_d[1]}")
