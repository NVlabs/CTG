import stlcg
from stlcg import Expression
import torch
import numpy as np
from matplotlib import pyplot as plt

class Box:
    def __init__(self, stop_sign_pos, stop_box_dim):
        '''
        stop_sign_pos: [B, 2]
        stop_box_dim: [B, 2]
        '''
        self.stop_sign_pos = stop_sign_pos
        self.stop_box_dim = stop_box_dim

        self.set_bounds()

    def update_pos_and_dim(self, stop_sign_pos=None, stop_box_dim=None):
        if stop_sign_pos is not None:
            self.stop_sign_pos = stop_sign_pos
        if stop_box_dim is not None:
            self.stop_box_dim = stop_box_dim
        self.set_bounds()
    
    def set_bounds(self):
        x_min = self.stop_sign_pos[...,0] - self.stop_box_dim[...,0]/2
        x_max = self.stop_sign_pos[...,0] + self.stop_box_dim[...,0]/2
        y_min = self.stop_sign_pos[...,1] - self.stop_box_dim[...,1]/2
        y_max = self.stop_sign_pos[...,1] + self.stop_box_dim[...,1]/2

        self.x_min = x_min[:, None, None]
        self.y_min = y_min[:, None, None]
        self.x_max = x_max[:, None, None]
        self.y_max = y_max[:, None, None]
        # print('self.x_min.shape', self.x_min.shape)
    def __str__(self):
        return 'stop_sign_pos ' + self.stop_sign_pos.__str__() + '\n stop_box_dim ' + self.stop_box_dim.__str__()
        # return f'x: ({self.x_min} , {self.x_max}), y: ({self.y_min} , {self.y_max})'

    def __repr__(self):
        return self.__str__()


class StopSignRule():
    def __init__(self, stop_sign_pos, stop_box_dim, low_speed_th=0.1) -> None:
        self.horizon_length = None
        self.time_step_to_start = None
        self.num_time_steps_to_stop = None

        self.low_speed_th = low_speed_th

        self.stop_box = Box(stop_sign_pos, stop_box_dim)
    
    def update_stop_box(self, stop_sign_pos=None, stop_box_dim=None):
        '''
        stop_sign_pos: tensor [B, 2]
        stop_box_dim: tensor [B, 2]
        '''
        self.stop_box.update_pos_and_dim(stop_sign_pos, stop_box_dim)
    
    @torch.no_grad()
    def check_inclusion(self, pos):
        '''
        -pos: [B*N, T, 1, 2]
        -in_stop_box: [B*N, T, 1]
        '''
        assert len(pos.shape) == 4
        x_check = ((pos[..., 0] >= self.stop_box.x_min) & (pos[..., 0] <= self.stop_box.x_max))
        y_check = ((pos[..., 1] >= self.stop_box.y_min) & (pos[..., 1] <= self.stop_box.y_max))
        in_stop_box = x_check & y_check
        # print('pos[..., 0].shape, self.stop_box.x_min.shape, x_check.shape', pos[..., 0].shape, self.stop_box.x_min.shape, x_check.shape)
        return in_stop_box

    def set_stl_rules(self, horizon_length, time_step_to_start, num_time_steps_to_stop, use_until=False):
        '''
        Always([in stop sign region] -> Eventually( Always_[0, 3](low speed And in stop region) )
        <=> Always( Not(in stop region) Or Eventually( Always_[0, 3](low speed And in stop region) )
        '''
        reversed = True
        speed = Expression('speed', [], reversed)
        pos_x = Expression('pos_x', [], reversed)
        pos_y = Expression('pos_y', [], reversed)
        
        already_stopped = Expression('already_stopped', [], reversed)

        # Check if the car ever enters the stop sign box
        x_check = ((pos_x >= self.stop_box.x_min) & (pos_x <= self.stop_box.x_max))
        y_check = ((pos_y >= self.stop_box.y_min) & (pos_y <= self.stop_box.y_max))
        in_stop_box = x_check & y_check
        
        out_stop_box = stlcg.Negation(in_stop_box)

        low_speed = ((speed >= -self.low_speed_th) & (speed <= self.low_speed_th))
        middle_speed = stlcg.Negation(low_speed)
        # high_speed = middle_speed
        high_speed = ((speed < -self.low_speed_th) | (speed > self.low_speed_th))

        have_stopped = (already_stopped == 1)
        have_not_stopped = stlcg.Negation(have_stopped)
        

        in_stop_box_and_have_not_stopped = in_stop_box & have_not_stopped

        # Check if the car comes to an almost halt when within the stop sign box
        # stop = stlcg.Eventually(subformula =((speed >= -0.1) & (speed <= 0.1)) & in_stop_box)
        
        stop = stlcg.Eventually(stlcg.Always(in_stop_box & low_speed, \
                                               interval=[0, num_time_steps_to_stop]), interval=[time_step_to_start, horizon_length])
        

        if use_until:
            # The car never enters the stop sign box until it halts for some time steps
            self.stl = stlcg.Until(subformula1=in_stop_box_and_have_not_stopped, subformula2=stop)
        else:
            # The car always either outside the stop sign box or the stop in the stop sign box
            # in_stop_box_stop = stlcg.Always(stlcg.Implies(subformula1=in_stop_box_and_have_not_stopped, subformula2=stop))
            in_stop_box_stop = stlcg.Always(stlcg.Implies(subformula1=in_stop_box, subformula2=stop))
            
            out_stop_box_middle_speed = (out_stop_box & middle_speed)

            have_stopped_high_speed = (have_stopped & high_speed)

            in_and_out_speed = (in_stop_box_stop | out_stop_box_middle_speed)

            self.stl = in_and_out_speed #| have_stopped_high_speed

    def _stl_signal(self, speed, pos_x, pos_y, already_stopped):
        '''
        arrange the signals according to the ordering prescribed by the stl formula
        '''
        stop_box_signal = ((pos_x, pos_x), (pos_y, pos_y))
        speed_signal = (speed, speed)

        box_speed_signal = (stop_box_signal, speed_signal)

        stop_signal = already_stopped

        first = (stop_box_signal, box_speed_signal)
        # first = ((stop_box_signal, stop_signal), box_speed_signal)
        second = box_speed_signal
        
        
        third = (stop_signal, speed_signal)

        # return ((first, second), third)
        return (first, second)

    def _shape_signal_batch(self, speed, pos_x, pos_y, already_stopped):
        '''
        speed: [B*N, T]
        pos_x: [B*N, T]
        pos_y: [B*N, T]
        already_stopped: [B*N, T]
        rearrange the signals to match the shape required by stlcg
        '''
        speed = speed.flip(1).view([speed.shape[0], speed.shape[1], 1])
        pos_x = pos_x.flip(1).view([pos_x.shape[0], pos_x.shape[1], 1])
        pos_y = pos_y.flip(1).view([pos_y.shape[0], pos_y.shape[1], 1])
        already_stopped = already_stopped.flip(1).view([already_stopped.shape[0], already_stopped.shape[1], 1])

        return self._stl_signal(speed, pos_x, pos_y, already_stopped)

    def get_robustness(self, speed, pos_x, pos_y, already_stopped, horizon_length, time_step_to_start, num_time_steps_to_stop, scale=-1, use_until=False):
        '''
        return stl robustness for the passed signal
        '''
        if self.horizon_length is None \
           or horizon_length != self.horizon_length \
           or time_step_to_start != self.time_step_to_start \
           or num_time_steps_to_stop != self.num_time_steps_to_stop:
            self.horizon_length = horizon_length
            self.time_step_to_start = time_step_to_start
            self.num_time_steps_to_stop = num_time_steps_to_stop

            # self.set_stl_rules(horizon_length, time_step_to_start, num_time_steps_to_stop)
            self.set_stl_rules(horizon_length, time_step_to_start, num_time_steps_to_stop, use_until=use_until)
            
        # print(speed.shape, pos_x.shape, pos_y.shape)
        return self.stl.robustness(
            self._shape_signal_batch(speed, pos_x, pos_y, already_stopped), 
            scale=scale
        ).squeeze()
        


if __name__=="__main__":

    # 'example', 'check_robustness'
    mode = 'example'
    if mode == 'example':
        scale = 20.0 # 20.0
        iter_num = 50
        lr = 0.001
        horizon_length = 20
        time_step_to_start = 10
        num_time_steps_to_stop = 5

        # Dummy example setup
        stop_sign_pos = torch.tensor([[8., 0.]]*2)
        stop_box_dim = torch.tensor([[2., 2.]]*2)
        initial_v = 0.5
        initial_acc = 0.0

        t = torch.arange(0, horizon_length, 1)
        t = torch.stack([t, t])

        # diagonal matrix
        mat = torch.ones(t.shape[1], t.shape[1])
        mat = torch.tril(mat, diagonal=0)
        mat = mat.repeat(t.shape[0],1,1)

        acc = initial_acc*torch.ones_like(t)
        acc = torch.tensor(acc, requires_grad=True)

        robustness_list = []

        optimizer = torch.optim.Adam([acc], lr=lr)

        for _ in range(iter_num):
            # Estimate speed, x and y
            # print('mat.shape', mat.shape)
            # print('acc.unsqueeze(-1).shape', acc.unsqueeze(-1).shape)
            speed = initial_v + torch.bmm(mat, acc.unsqueeze(-1))
            # print('speed.shape', speed.shape)
            x = torch.bmm(mat, speed)
            y = 0.0*t

            # Visualize example
            # stop_box_x_min = stop_sign_pos[0] - stop_box_dim[0]/2
            # stop_box_x_max = stop_sign_pos[0] + stop_box_dim[0]/2
            # stop_box_y_min = stop_sign_pos[1] - stop_box_dim[1]/2
            # stop_box_y_max = stop_sign_pos[1] + stop_box_dim[1]/2
            
            # plt.plot(
            #     [stop_box_x_min, stop_box_x_min, stop_box_x_max, stop_box_x_max, stop_box_x_min],
            #     [stop_box_y_min, stop_box_y_max, stop_box_y_max, stop_box_y_min, stop_box_y_min],
            # )
            # plt.plot(x.detach().numpy()[0,...], y.detach().numpy()[0,...])

            # Compute robustness of the signal
            stopSign = StopSignRule(stop_sign_pos, stop_box_dim)
            robustness = stopSign.get_robustness(speed.squeeze(), x.squeeze(), y, horizon_length, \
                        time_step_to_start, num_time_steps_to_stop, scale=scale, use_until=False)

            # Display results
            print('robustness[0]', robustness[0])
            robustness_list.append(robustness[0].detach().cpu().numpy())
            
            # plt.show()

            # Gradient descent
            
            loss = -torch.clip(robustness, max=0)
            loss = torch.mean(loss)
            loss.backward()
            # print('x.grad', x.grad)
            # print('y.grad', y.grad)
            print('x[0]', x[0])
            print('speed[0]', speed[0])
            print('acc[0]', acc[0])
            print('acc.grad[0]', acc.grad[0])

            # speed.data = speed.data - step_size * speed.grad
            optimizer.step()
            optimizer.zero_grad()


        plt.plot(np.arange(len(robustness_list)), robustness_list)
        plt.xlabel('iteration')
        plt.ylabel('robustness')
        plt.show()
    elif mode == 'check_robustness':
        speed = torch.tensor([[ 6.8248,  6.8392,  6.8454,  6.8454,  6.8504,  6.8077,  6.8070,  6.8063,
          6.8023,  6.8034,  6.8012,  6.8068,  6.8027,  6.8090,  6.8081,  6.8201,
          6.8231,  6.8282,  6.8373,  6.8451],
        [ 6.0692,  6.0855,  6.0931,  6.1224,  6.1318,  6.1438,  6.1468,  6.1501,
          6.1496,  6.1602,  6.1605,  6.1704,  6.1701,  6.2039,  6.2034,  6.2006,
          6.2002,  6.1973,  6.0971,  6.1043],
        [-0.0141, -0.0120, -0.0112, -0.0104, -0.0119, -0.0092, -0.0142, -0.0109,
         -0.0123, -0.0096, -0.0115, -0.0096, -0.0138, -0.0138, -0.0188, -0.0184,
         -0.0199, -0.0185, -0.0185, -0.0177],
        [-0.0344, -0.0366, -0.0429, -0.0457, -0.0506, -0.0522, -0.0593, -0.0611,
         -0.0647, -0.0655, -0.0694, -0.0692, -0.0758, -0.0763, -0.0825, -0.0844,
         -0.0875, -0.0886, -0.0884, -0.0914]], device='cuda:0')
        pos_x = torch.tensor([[ 6.7773e-01,  1.3608e+00,  2.0449e+00,  2.7295e+00,  3.4141e+00,
          4.0967e+00,  4.7773e+00,  5.4580e+00,  6.1382e+00,  6.8184e+00,
          7.4985e+00,  8.1787e+00,  8.8589e+00,  9.5396e+00,  1.0220e+01,
          1.0901e+01,  1.1583e+01,  1.2266e+01,  1.2949e+01,  1.3633e+01],
        [ 5.7764e-01,  1.1855e+00,  1.7944e+00,  2.4053e+00,  3.0181e+00,
          3.6318e+00,  4.2466e+00,  4.8613e+00,  5.4761e+00,  6.0913e+00,
          6.7075e+00,  7.3242e+00,  7.9414e+00,  8.5601e+00,  9.1802e+00,
          9.8003e+00,  1.0420e+01,  1.1040e+01,  1.1655e+01,  1.2265e+01],
        [-4.6086e-04, -1.7636e-03, -2.9204e-03, -3.9971e-03, -5.1119e-03,
         -6.1667e-03, -7.3340e-03, -8.5881e-03, -9.7477e-03, -1.0842e-02,
         -1.1895e-02, -1.2951e-02, -1.4123e-02, -1.5506e-02, -1.7139e-02,
         -1.9001e-02, -2.0915e-02, -2.2836e-02, -2.4686e-02, -2.6493e-02],
        [-3.6979e-04, -3.9213e-03, -7.9000e-03, -1.2333e-02, -1.7151e-02,
         -2.2289e-02, -2.7866e-02, -3.3890e-02, -4.0180e-02, -4.6688e-02,
         -5.3432e-02, -6.0364e-02, -6.7615e-02, -7.5218e-02, -8.3160e-02,
         -9.1507e-02, -1.0010e-01, -1.0890e-01, -1.1775e-01, -1.2675e-01]],
       device='cuda:0')
        pos_y = torch.tensor([[ 0.0000e+00,  2.6455e-03,  7.4635e-03,  1.4872e-02,  2.4538e-02,
          3.6219e-02,  4.8967e-02,  6.2876e-02,  7.7425e-02,  9.2157e-02,
          1.0699e-01,  1.2187e-01,  1.3633e-01,  1.5059e-01,  1.6449e-01,
          1.7789e-01,  1.9059e-01,  2.0274e-01,  2.1400e-01,  2.2449e-01],
        [ 0.0000e+00, -9.4748e-04, -2.8033e-03, -5.2485e-03, -8.3842e-03,
         -1.2041e-02, -1.6359e-02, -2.1184e-02, -2.6563e-02, -3.2129e-02,
         -3.8038e-02, -4.3905e-02, -4.9940e-02, -5.5803e-02, -6.1430e-02,
         -6.6881e-02, -7.2107e-02, -7.6974e-02, -8.1354e-02, -8.5348e-02],
        [ 0.0000e+00,  1.3551e-07,  4.4378e-07,  8.6054e-07,  1.3625e-06,
          1.8543e-06,  2.5621e-06,  3.3253e-06,  4.1276e-06,  4.9467e-06,
          5.9200e-06,  6.9463e-06,  8.2054e-06,  9.8743e-06,  1.2072e-05,
          1.4833e-05,  1.8007e-05,  2.1421e-05,  2.5187e-05,  2.8974e-05],
        [ 0.0000e+00, -5.6438e-07, -1.6876e-06, -3.1609e-06, -5.4631e-06,
         -8.4899e-06, -1.1859e-05, -1.5980e-05, -2.0409e-05, -2.5047e-05,
         -2.9417e-05, -3.4029e-05, -3.8518e-05, -4.3107e-05, -4.7566e-05,
         -5.2338e-05, -5.7140e-05, -6.2687e-05, -6.7482e-05, -7.1732e-05]],
       device='cuda:0')

        stop_sign_pos = torch.tensor([[15.5136,  0.1324],
        [19.4990,  1.1519],
        [-0.2129,  0.0308],
        [ 0.0601, -0.0562]], device='cuda:0')

        stop_box_dim = torch.tensor([[10., 10.],
        [10., 10.],
        [10., 10.],
        [10., 10.]], device='cuda:0')

        horizon_length = 20
        time_step_to_start = 0
        num_time_steps_to_stop = 1
        scale = 20

        stopSign = StopSignRule(stop_sign_pos, stop_box_dim)
        robustness = stopSign.get_robustness(speed, pos_x, pos_y, horizon_length, \
                    time_step_to_start, num_time_steps_to_stop, scale=scale, use_until=False)
        print('robustness', robustness)


