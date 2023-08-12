import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import MAU
# from core.models import STAU
# from core.models import AAU
# from core.models import STAUv2
# from core.models import AAUv2
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
            # 'stau': STAU.RNN,
            # 'aau': AAU.RNN,
            # 'stauv2': STAUv2.RNN,
            # 'aauv2': AAUv2.RNN
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
        self.L1_loss = nn.L1Loss()

    def save(self, itr):
        stats = {'net_param': self.network.state_dict()}
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)

    def load(self, pm_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

    def train(self, data, data_mask, data_back, mask, itr):
        frames = data
        frames_mask = data_mask
        frames_back = data_back
        # 开启训练模式
        self.network.train()
        # 将数据转换成tensor并转存到device中 即，GPU中
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        frames_mask_tensor = torch.FloatTensor(frames_mask).to(self.configs.device)
        frames_back_tensor = torch.FloatTensor(frames_back).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        # 进入搭建network进行数据训练
        # 输入的是
        # 1.frames_tensor 图片信息 16 * 20 * 1 * 64 * 64
        # 2. mask_tensor real_input_flag掩码信息 16 * 9 * 64 * 64 * 1
        # 输出的是
        # next_frames 16 * 19 * 1 * 64 * 64
        next_frames, next_frames_mask, next_frames_back = self.network(frames_tensor,frames_mask_tensor,frames_back_tensor, mask_tensor,itr)
        ground_truth = frames_tensor
        ground_truth_mask = frames_mask_tensor
        ground_truth_back = frames_back_tensor

        batch_size = next_frames.shape[0]
        self.optimizer.zero_grad()
        loss_l1 = self.L1_loss(next_frames,
                               ground_truth[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames,
                                     ground_truth[:, 1:])
        loss_l2_mask = self.MSE_criterion(next_frames_mask,
                                     ground_truth_mask[:, 1:])
        loss_l2_back = self.MSE_criterion(next_frames_back,
                                      ground_truth_back[:, 1:])

        total = loss_l2 + loss_l2_mask + loss_l2_back

        num_a = total / loss_l2
        num_b = total / loss_l2_mask
        num_c = total / loss_l2_back

        print("num_a: ", num_a)
        print("num_b: ", num_b)
        print("num_c: ", num_c)
        
        loss_gen = 1.5 * num_a * loss_l2 + num_b * loss_l2_mask + num_c * loss_l2_back
        
        print("1.5 * num_a * loss_l2: ", 1.5 * num_a * loss_l2)
        print("num_b * loss_l2_mask: ", num_b * loss_l2_mask)
        print("num_c * loss_l2_back: ", num_c * loss_l2_back)
        print("loss_gen: ", loss_gen)
        loss_gen.backward()
        self.optimizer.step()

        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            # self.scheduler_F.step()
            # self.scheduler_D.step()
            print('Lr decay to:%.8f', self.optimizer.param_groups[0]['lr'])
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy()

    def test(self, data,data_mask, data_back, mask,itr):
        frames = data
        frames_mask = data_mask
        frames_back = data_back
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        frames_mask_tensor = torch.FloatTensor(frames_mask).to(self.configs.device)
        frames_back_tensor = torch.FloatTensor(frames_back).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor,frames_mask_tensor,frames_back_tensor,mask_tensor,itr)
        return next_frames.detach().cpu().numpy()
