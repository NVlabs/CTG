from tbsim.dynamics.base import DynType, Dynamics
import torch
import numpy as np
from copy import deepcopy


class Unicycle(Dynamics):
    def __init__(
        self, name, max_steer=0.5, max_yawvel=8, acce_bound=[-6, 4], vbound=[-10, 30]
    ):
        self._name = name
        self._type = DynType.UNICYCLE
        self.xdim = 4
        self.udim = 2
        self.cyclic_state = [3]
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.max_steer = max_steer
        self.max_yawvel = max_yawvel

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            theta = x[..., 3:4]
            dxdt = np.hstack(
                (np.cos(theta) * x[..., 2:3], np.sin(theta) * x[..., 2:3], u)
            )
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            theta = x[..., 3:4]
            dxdt = torch.cat(
                (torch.cos(theta) * x[..., 2:3],
                 torch.sin(theta) * x[..., 2:3], u),
                dim=-1,
            )
        else:
            raise NotImplementedError
        return dxdt

    def step(self, x, u, dt, bound=True):
        assert x.shape[:-1] == u.shape[:-1]
        # print('x.shape:', x.shape, 'u.shape:', u.shape)
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)

            theta = x[..., 3:4]
            dxdt = np.hstack(
                (
                    np.cos(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    np.sin(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    u,
                )
            )
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            # print('original before clip u[0]', u[0])
            if bound:
                lb, ub = self.ubound(x)
                # s = (u - lb) / torch.clip(ub - lb, min=1e-3)
                # u = lb + (ub - lb) * torch.sigmoid(s)
                u = torch.clip(u, lb, ub)
            # print('original after clip u[0]', u[0])
            theta = x[..., 3:4]
            dxdt = torch.cat(
                (
                    torch.cos(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    torch.sin(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    u,
                ),
                dim=-1,
            )
        else:
            raise NotImplementedError
        # print("x.size()", x.size())
        # print('x[1]', 'dxdt[1]*dt')
        # print(x[1], dxdt[1]*dt)
        return x + dxdt * dt

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if isinstance(x, np.ndarray):
            v = x[..., 2:3]
            yawbound = np.minimum(
                self.max_steer * v,
                self.max_yawvel / np.clip(np.abs(v), a_min=0.1, a_max=None),
            )
            acce_lb = np.clip(
                np.clip(self.vbound[0] - v, None, self.acce_bound[1]),
                self.acce_bound[0],
                None,
            )
            acce_ub = np.clip(
                np.clip(self.vbound[1] - v, self.acce_bound[0], None),
                None,
                self.acce_bound[1],
            )
            lb = np.hstack((acce_lb, -yawbound))
            ub = np.hstack((acce_ub, yawbound))
            return lb, ub
        elif isinstance(x, torch.Tensor):
            v = x[..., 2:3]
            yawbound = torch.minimum(
                self.max_steer * torch.abs(v),
                self.max_yawvel / torch.clip(torch.abs(v), min=0.1),
            )
            yawbound = torch.clip(yawbound, min=0.1)
            acce_lb = torch.clip(
                torch.clip(self.vbound[0] - v, max=self.acce_bound[1]),
                min=self.acce_bound[0],
            )
            acce_ub = torch.clip(
                torch.clip(self.vbound[1] - v, min=self.acce_bound[0]),
                max=self.acce_bound[1],
            )
            lb = torch.cat((acce_lb, -yawbound), dim=-1)
            ub = torch.cat((acce_ub, yawbound), dim=-1)
            return lb, ub

        else:
            raise NotImplementedError

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2yaw(x):
        return x[..., 3:]

    @staticmethod
    def calculate_vel(pos, yaw, dt, mask):
        if isinstance(pos, torch.Tensor):
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * torch.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * torch.sin(
                yaw[..., 1:, :]
            )
            # right finite difference velocity
            vel_r = torch.cat((vel[..., 0:1, :], vel), dim=-2)
            # left finite difference velocity
            vel_l = torch.cat((vel, vel[..., -1:, :]), dim=-2)
            mask_r = torch.roll(mask, 1, dims=-1)
            mask_r[..., 0] = False
            mask_r = mask_r & mask

            mask_l = torch.roll(mask, -1, dims=-1)
            mask_l[..., -1] = False
            mask_l = mask_l & mask
            vel = (
                (mask_l & mask_r).unsqueeze(-1) * (vel_r + vel_l) / 2
                + (mask_l & (~mask_r)).unsqueeze(-1) * vel_l
                + (mask_r & (~mask_l)).unsqueeze(-1) * vel_r
            )
        elif isinstance(pos, np.ndarray):
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * np.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * np.sin(yaw[..., 1:, :])
            # right finite difference velocity
            vel_r = np.concatenate((vel[..., 0:1, :], vel), axis=-2)
            # left finite difference velocity
            vel_l = np.concatenate((vel, vel[..., -1:, :]), axis=-2)
            mask_r = np.roll(mask, 1, axis=-1)
            mask_r[..., 0] = False
            mask_r = mask_r & mask
            mask_l = np.roll(mask, -1, axis=-1)
            mask_l[..., -1] = False
            mask_l = mask_l & mask
            vel = (
                np.expand_dims(mask_l & mask_r,-1) * (vel_r + vel_l) / 2
                + np.expand_dims(mask_l & (~mask_r),-1) * vel_l
                + np.expand_dims(mask_r & (~mask_l),-1) * vel_r
            )
        else:
            raise NotImplementedError
        return vel
    @staticmethod
    def inverse_dyn(x,xp,dt):
        return (xp[...,2:]-x[...,2:])/dt