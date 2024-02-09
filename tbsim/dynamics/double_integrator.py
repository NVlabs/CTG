from tbsim.dynamics.base import DynType, Dynamics
import torch
import numpy as np
from copy import deepcopy



class DoubleIntegrator(Dynamics):
    def __init__(self, name, abound, vbound=None):
        self._name = name
        self._type = DynType.DI
        self.xdim = abound.shape[0] * 2
        self.udim = abound.shape[0]
        self.cyclic_state = list()
        self.vbound = np.array(vbound)
        self.abound = np.array(abound)

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]
        if isinstance(x, np.ndarray):
            return np.hstack((x[..., 2:], u))
        elif isinstance(x, torch.Tensor):
            return torch.cat((x[..., 2:], u), dim=-1)
        else:
            raise NotImplementedError

    def step(self, x, u, dt, bound=True):

        if isinstance(x, np.ndarray):
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)
            xn = np.hstack(
                ((x[..., 2:4] + 0.5 * u * dt) * dt + x[..., 0:2], x[..., 2:4] + u * dt)
            )
        elif isinstance(x, torch.Tensor):
            if bound:
                lb, ub = self.ubound(x)
                u = torch.clip(u, min=lb, max=ub)
            xn = torch.clone(x)
            xn[..., 0:2] += (x[..., 2:4] + 0.5 * u * dt) * dt
            xn[..., 2:4] += u * dt
        else:
            raise NotImplementedError
        return xn

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if self.vbound is None:
            if isinstance(x, np.ndarray):
                lb = np.ones_like(x[..., 2:]) * self.abound[:, 0]
                ub = np.ones_like(x[..., 2:]) * self.abound[:, 1]

            elif isinstance(x, torch.Tensor):
                lb = torch.ones_like(x[..., 2:]) * torch.from_numpy(
                    self.abound[:, 0]
                ).to(x.device)
                ub = torch.ones_like(x[..., 2:]) * torch.from_numpy(
                    self.abound[:, 1]
                ).to(x.device)

            else:
                raise NotImplementedError
        else:
            if isinstance(x, np.ndarray):
                lb = (x[..., 2:] > self.vbound[:, 0]) * self.abound[:, 0]
                ub = (x[..., 2:] < self.vbound[:, 1]) * self.abound[:, 1]

            elif isinstance(x, torch.Tensor):
                lb = (
                    x[..., 2:] > torch.from_numpy(self.vbound[:, 0]).to(x.device)
                ) * torch.from_numpy(self.abound[:, 0]).to(x.device)
                ub = (
                    x[..., 2:] < torch.from_numpy(self.vbound[:, 1]).to(x.device)
                ) * torch.from_numpy(self.abound[:, 1]).to(x.device)
            else:
                raise NotImplementedError
        return lb, ub

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2yaw(x):
        # return torch.atan2(x[..., 3:], x[..., 2:3])
        return torch.zeros_like(x[..., 0:1])
