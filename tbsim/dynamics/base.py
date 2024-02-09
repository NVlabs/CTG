import torch
import numpy as np
import math, copy, time
import abc
from copy import deepcopy


class DynType:
    """
    Holds environment types - one per environment class.
    These act as identifiers for different environments.
    """

    UNICYCLE = 1
    SI = 2
    DI = 3
    BICYCLE = 4


class Dynamics(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name, **kwargs):
        self._name = name
        self.xdim = 4
        self.udim = 2

    @abc.abstractmethod
    def __call__(self, x, u):
        return

    @abc.abstractmethod
    def step(self, x, u, dt, bound=True):
        return

    @abc.abstractmethod
    def name(self):
        return self._name

    @abc.abstractmethod
    def type(self):
        return

    @abc.abstractmethod
    def ubound(self, x):
        return

    @staticmethod
    def state2pos(x):
        return

    @staticmethod
    def state2yaw(x):
        return


def forward_dynamics(
    dyn_model: Dynamics,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    step_time: float,
):
    """
    Integrate the state forward with initial state x0, action u
    Args:
        dyn_model (dynamics.Dynamics): dynamics model
        initial_states (Torch.tensor): state tensor of size [B, (A), 4]
        actions (Torch.tensor): action tensor of size [B, (A), T, 2]
        step_time (float): delta time between steps
    Returns:
        state tensor of size [B, (A), T, 4]
    """
    num_steps = actions.shape[-2]
    x = [initial_states] + [None] * num_steps
    for t in range(num_steps):
        x[t + 1] = dyn_model.step(x[t], actions[..., t, :], step_time)

    x = torch.stack(x[1:], dim=-2)
    pos = dyn_model.state2pos(x)
    yaw = dyn_model.state2yaw(x)
    return x, pos, yaw
