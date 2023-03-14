import os
import torch
import torch.nn as nn
from torch.optim import Adam

from core.data_provider.losses import KLLoss
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
        self.kl_loss = KLLoss()
        self.L1_loss = nn.L1Loss()
        self.beta = 1000

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

        next_frames, out_dict = self.network(frames_tensor, mask_tensor)
        ground_truth = frames_tensor

        batch_size = next_frames.shape[0]

        self.optimizer.zero_grad()

        kl_loss = self.kl_loss(
            mu1=out_dict["mu_post"], logvar1=out_dict["logvar_post"],
            mu2=out_dict["mu_prior"], logvar2=out_dict["logvar_prior"]
        )

        loss_l1 = self.L1_loss(next_frames,
                               ground_truth[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames,
                                     ground_truth[:, 1:])
        if kl_loss.item() > 1000000000:
            self.beta = 0.00000000001
        elif kl_loss.item() > 100000000:
            self.beta = 0.0000000001
        elif kl_loss.item() > 10000000:
            self.beta = 0.000000001
        elif kl_loss.item() > 1000000:
            self.beta = 0.00000001
        elif kl_loss.item() > 100000:
            self.beta = 0.0000001
        elif kl_loss.item() > 10000:
            self.beta = 0.000001
        elif kl_loss.item() > 1000:
            self.beta = 0.00001
        elif kl_loss.item() > 100:
            self.beta = 0.0001
        elif kl_loss.item() > 10:
            self.beta = 0.001
        elif kl_loss.item() > 1:
            self.beta = 0.01
        elif kl_loss.item() > 0.01:
            self.beta = 1
        elif kl_loss.item() > 0.001:
            self.beta = 1
        elif kl_loss.item() > 0.0001:
            self.beta = 10
        elif kl_loss.item() > 0.00001:
            self.beta = 100
        elif kl_loss.item() > 0.000001:
            self.beta = 1000
        else:
            self.beta = 1000
        print("kl_loss.item : ", kl_loss.item())
        print("current beta: ", self.beta)
        print("loss_l2: ", loss_l2)
        print("beta * kl_loss: ", self.beta * kl_loss)
        loss_gen = loss_l2 + self.beta * kl_loss
        print("loss_gen: ", loss_gen)
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
        next_frames, out_dict = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
