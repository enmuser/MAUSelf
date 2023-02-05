import torch
import torch.nn as nn
import math

from core.models.ConvLSTM import ConvLSTM
from core.models.DCGAN_Conv import dcgan_upconv


class MAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode,device):
        super(MAUCell, self).__init__()
        # in_channel = 1 ,num_hidden[i] = 64,  height = 16 , width = 16,
        # filter_size = (5,5), stride = 1,tau = 5,cell_mode = normal
        # num_hidden = 64
        self.num_hidden = num_hidden
        # padding = (2,2)
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        # d = 64 * 16 * 16 = 16384
        self.d = [128 * 16 * 16, 256 * 8 * 8, 512 * 4 * 4]
        # tau = 5
        self.tau = tau
        self.states = ['residual', 'normal']
        self.num_hidden_split = [128, 256, 512]
        self.device = device
        if not self.cell_mode in self.states:
            raise AssertionError
        conv_t_list = []
        conv_t_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 2, 3 * num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding,),
            nn.LayerNorm([3 * num_hidden * 2, height, width])
        ))
        conv_t_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 4, 3 * num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([3 * num_hidden * 4, height//2, width//2])
        ))
        conv_t_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 8, 3 * num_hidden * 8, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([3 * num_hidden * 8, height//4, width//4])
        ))
        self.conv_t = nn.ModuleList(conv_t_list)
        conv_t_next_list = []
        conv_t_next_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 2, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding,),
            nn.LayerNorm([num_hidden * 2, height, width])
        ))
        conv_t_next_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 4, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([num_hidden * 4, height//2, width//2])
        ))
        conv_t_next_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 8, num_hidden * 8, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([num_hidden * 8, height//4, width//4])
        ))
        self.conv_t_next = nn.ModuleList(conv_t_next_list)
        conv_s_list = []
        conv_s_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 2, 3 * num_hidden * 2, kernel_size=filter_size, stride=stride,
                      padding=self.padding, ),
            nn.LayerNorm([3 * num_hidden * 2, height, width])
        ))
        conv_s_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 4, 3 * num_hidden * 4, kernel_size=filter_size, stride=stride,
                      padding=self.padding, ),
            nn.LayerNorm([3 * num_hidden * 4, height//2, width//2])
        ))
        conv_s_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 8, 3 * num_hidden * 8, kernel_size=filter_size, stride=stride,
                      padding=self.padding, ),
            nn.LayerNorm([3 * num_hidden * 8, height//4, width//4])
        ))
        self.conv_s = nn.ModuleList(conv_s_list)

        conv_s_next_list = []
        conv_s_next_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 2, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([num_hidden * 2, height, width])
        ))
        conv_s_next_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 4, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([num_hidden * 4, height//2, width//2])
        ))
        conv_s_next_list.append(nn.Sequential(
            nn.Conv2d(in_channel * 8, num_hidden * 8, kernel_size=filter_size, stride=stride, padding=self.padding, ),
            nn.LayerNorm([num_hidden * 8, height//4, width//4])
        ))
        self.conv_s_next = nn.ModuleList(conv_s_next_list)

        conv_t_lower_level_list = []
        conv_t_lower_level_list.append(dcgan_upconv(512, 256))
        conv_t_lower_level_list.append(dcgan_upconv(256, 128))
        self.conv_t_lower = nn.ModuleList(conv_t_lower_level_list)
        conv_s_lower_level_list = []
        conv_s_lower_level_list.append(dcgan_upconv(512,256))
        conv_s_lower_level_list.append(dcgan_upconv(256,128))
        self.conv_s_lower = nn.ModuleList(conv_s_lower_level_list)

        conv_t_1x1_list = []
        conv_t_1x1_list.append(nn.Sequential(nn.Conv2d(256, 256, 1)))
        conv_t_1x1_list.append(nn.Sequential(nn.Conv2d(128, 128, 1)))
        self.conv_t_1x1 = nn.ModuleList(conv_t_1x1_list)

        conv_s_1x1_list = []
        conv_s_1x1_list.append(nn.Sequential(nn.Conv2d(256, 256, 1)))
        conv_s_1x1_list.append(nn.Sequential(nn.Conv2d(128, 128, 1)))
        self.conv_s_1x1 = nn.ModuleList(conv_s_1x1_list)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        # T_t => T(k,t-1) 当前时间特征
        # S_t => S(k-1,t) 当前空间特征
        # t_att => T(k,t-tau:t-1)
        # s_att => S(k-1,t-tau:t-1)
        T_new_return = []
        S_new_return = []
        for index in reversed(range(0, 3)):
            current_t_att = []
            for i in range(self.tau):
                current_t_att.append(t_att[i][index])
            current_t_att = torch.stack(current_t_att, dim=0)
            # 一次空间特征卷积操作
            s_next = self.conv_s_next[index](S_t[index])
            # 一次时间特征卷积操作
            t_next = self.conv_t_next[index](T_t[index])
            # 计算注意分数权重
            weights_list = []
            for i in range(self.tau):
                # tau = τ = 5
                # qi的计算 当前空间特征卷积操作的结果 与 历史前τ个进行Hadamard乘积
                weights_list.append((s_att[i][index] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d[index]))
            weights_list = torch.stack(weights_list, dim=0)
            weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
            weights_list = self.softmax(weights_list)
            T_trend = current_t_att * weights_list
            # T_trend = T_att 长期运动信息
            T_trend = T_trend.sum(dim=0)
            # t_att_gate = Uf 融合门
            t_att_gate = torch.sigmoid(t_next)
            # T_fusion = T_AMI
            # 表示增强的运动信息 长期运动信息 T_trend 和 短期运动信息 T_t 进行融合得到
            T_fusion = T_t[index] * t_att_gate + (1 - t_att_gate) * T_trend
            # T_AMI 卷积一次 => U_t   T_concat shape=16 * 192 * 16 * 16
            T_concat = self.conv_t[index](T_fusion)
            # S_t 卷积一次 => U_s   S_concat shape=16 * 192 * 16 * 16
            S_concat = self.conv_s[index](S_t[index])
            # T_concat 一分为三 t_g, t_t, t_s shape= 16 * 64 * 16 * 16
            t_g, t_t, t_s = torch.split(T_concat, self.num_hidden_split[index], dim=1)
            # S_concat 一分为三 s_g, s_t, s_s shape= 16 * 64 * 16 * 16
            s_g, s_t, s_s = torch.split(S_concat, self.num_hidden_split[index], dim=1)
            # T_gate 为 U_t_1 第一分组
            T_gate = torch.sigmoid(t_g)
            # S_gate 为 U_s_1 第一分组
            S_gate = torch.sigmoid(s_g)
            # sigmoid(U_t_1) * U_t_2 + (1-sigmoid(U_t_1))*U_s_2
            T_new = T_gate * t_t + (1 - T_gate) * s_t
            # sigmoid(U_s_1) * U_s_2 + (1-sigmoid(U_s_1))*U_t_2
            S_new = S_gate * s_s + (1 - S_gate) * t_s
            # 如果是残差网络 将 S_t 加到 S_new 上
            if self.cell_mode == 'residual':
                S_new = S_new + S_t[index]
            T_new_return.append(T_new)
            S_new_return.append(S_new)
        for i in range(0, 2):
            T_new_return_tmp = self.conv_t_lower[i](T_new_return[i])
            TT_gate = torch.sigmoid(T_new_return[i + 1])
            T_new_return[i + 1] = T_new_return[i + 1] * TT_gate + (1-TT_gate) * T_new_return_tmp
            T_new_return[i + 1] = self.conv_t_1x1[i](T_new_return[i + 1])
            S_new_return_tmp = self.conv_s_lower[i](S_new_return[i])
            SS_gate = torch.sigmoid(S_new_return[i + 1])
            S_new_return[i + 1] = S_new_return[i + 1] * SS_gate + (1 - SS_gate) * S_new_return_tmp
            S_new_return[i + 1] = self.conv_s_1x1[i](S_new_return[i + 1])
        T_new_return.reverse()
        S_new_return.reverse()
        return T_new_return, S_new_return
