import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam

from core.data_provider.losses import KLLoss
from core.models import MAU
import torch.optim.lr_scheduler as lr_scheduler

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = configs.num_layers
        networks_map = {
            'mau': MAU.RNN,
        }
        num_hidden = []
        for i in range(configs.num_layers):
            num_hidden.append(configs.num_hidden)
        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        # print("Network state:")
        # for param_tensor in self.network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', self.network.state_dict()[param_tensor].size())
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=configs.lr_decay)

        self.MSE_criterion = nn.MSELoss()
        self.kl_loss = KLLoss()
        self.L1_loss = nn.L1Loss()
        self.beta = 0.001

    def save(self, itr):
        stats = {'net_param': self.network.state_dict()}
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)

    def load(self, pm_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

    def train(self, data, mask, itr):
        frames = data
        self.network.train()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        next_frames = self.network(frames_tensor, mask_tensor)
        ground_truth = frames_tensor

        batch_size = next_frames.shape[0]

        self.optimizer.zero_grad()

        empty_lists = []

        # out_dict = {"slow_self": deepcopy(empty_lists), "slow_pre": deepcopy(empty_lists),
        #             "middle_self": deepcopy(empty_lists), "middle_pre": deepcopy(empty_lists),
        #             "fast_self": deepcopy(empty_lists), "fast_pre": deepcopy(empty_lists)}
        #
        # for slow in range(11, 20):
        #     slow_diff_frame = ground_truth[:, slow] - ground_truth[:, slow - 1]
        #     slow_diff_frame_pre = next_frames[:, slow - 1] - next_frames[:, slow - 2]
        #     out_dict["slow_self"].append(slow_diff_frame)
        #     out_dict["slow_pre"].append(slow_diff_frame_pre)
        # for middle in range(12, 20, 2):
        #     middle_diff_frame = ground_truth[:, middle] - ground_truth[:, middle - 2]
        #     middle_diff_frame_pre = next_frames[:, middle - 1] - next_frames[:, middle - 3]
        #     out_dict["middle_self"].append(middle_diff_frame)
        #     out_dict["middle_pre"].append(middle_diff_frame_pre)
        # for fast in range(13, 20, 3):
        #     fast_diff_frame = ground_truth[:, fast] - ground_truth[:, fast - 3]
        #     fast_diff_frame_pre = next_frames[:, fast - 1] - next_frames[:, fast - 4]
        #     out_dict["fast_self"].append(fast_diff_frame)
        #     out_dict["fast_pre"].append(fast_diff_frame_pre)
        #
        # kl_loss = self.kl_loss(
        #     slow=out_dict["slow_self"], slow_pre=out_dict["slow_pre"],
        #     middle=out_dict["middle_self"], middle_pre=out_dict["middle_pre"],
        #     fast=out_dict["fast_self"], fast_pre=out_dict["fast_pre"]
        # )

        loss_l1 = self.L1_loss(next_frames,
                               ground_truth[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames,
                                     ground_truth[:, 1:])
        #print("kl_loss : ", kl_loss.item())
        # print("self.beta: ", self.beta)
        # print("batch_size: ", batch_size)
        # print("self.beta * (kl_loss  / batch_size): ", self.beta * (kl_loss / batch_size))
        #print("loss_l2: ", loss_l2)
        # loss_gen = loss_l2 + 0.625 * kl_loss
        loss_gen = loss_l2
        print("loss_gen: ", loss_gen)
        # print("loss_gen = loss_l2 + 0.625 * kl_loss => ", (loss_l2 + 0.625 * kl_loss))
        loss_gen.backward()
        self.optimizer.step()

        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            # self.scheduler_F.step()
            # self.scheduler_D.step()
            print('Lr decay to:%.8f', self.optimizer.param_groups[0]['lr'])
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy()

    def test(self, data, mask):
        frames = data
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
