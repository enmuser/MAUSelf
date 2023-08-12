import torch
import torch.nn as nn
import math

class MAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()
        # in_channel = 1 ,num_hidden[i] = 64,  height = 16 , width = 16,
        # filter_size = (5,5), stride = 1,tau = 5,cell_mode = normal
        # num_hidden = 64
        self.num_hidden = num_hidden
        # padding = (2,2)
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        # d = 64 * 16 * 16 = 16384
        self.d = num_hidden * height * width
        # tau = 5
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_level_one = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_level_two = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_next_level_one = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_t_next_level_two = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_level_one = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_level_two = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_s_next_level_one = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_s_next_level_two = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, T_t_level_one, T_t_level_two, S_t, S_t_level_one, S_t_level_two, t_att, s_att, t_att_level_one, s_att_level_one, t_att_level_two, s_att_level_two):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)

        s_next_level_one = self.conv_s_next_level_one(S_t_level_one)
        t_next_level_one = self.conv_t_next_level_one(T_t_level_one)

        s_next_level_two = self.conv_s_next_level_two(S_t_level_two)
        t_next_level_two = self.conv_t_next_level_two(T_t_level_two)


        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)

        weights_list_level_one = []
        for i in range(self.tau):
            weights_list_level_one.append((s_att_level_one[i] * s_next_level_one).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_level_one = torch.stack(weights_list_level_one, dim=0)
        weights_list_level_one = torch.reshape(weights_list_level_one, (*weights_list_level_one.shape, 1, 1, 1))
        weights_list_level_one = self.softmax(weights_list_level_one)
        T_trend_level_one = t_att_level_one * weights_list_level_one
        T_trend_level_one = T_trend_level_one.sum(dim=0)


        weights_list_level_two = []
        for i in range(self.tau):
            weights_list_level_two.append((s_att_level_two[i] * s_next_level_two).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_level_two = torch.stack(weights_list_level_two, dim=0)
        weights_list_level_two = torch.reshape(weights_list_level_two, (*weights_list_level_two.shape, 1, 1, 1))
        weights_list_level_two = self.softmax(weights_list_level_two)
        T_trend_level_two = t_att_level_two * weights_list_level_two
        T_trend_level_two = T_trend_level_two.sum(dim=0)



        t_att_gate_level_one = torch.sigmoid(t_next_level_one)
        T_fusion_level_one = T_t_level_one * t_att_gate_level_one + (1 - t_att_gate_level_one) * T_trend_level_one
        T_concat_level_one = self.conv_t_level_one(T_fusion_level_one)
        S_concat_level_one = self.conv_s_level_one(S_t_level_one)
        t_g_level_one, t_t_level_one, t_s_level_one = torch.split(T_concat_level_one, self.num_hidden, dim=1)
        s_g_level_one, s_t_level_one, s_s_level_one = torch.split(S_concat_level_one, self.num_hidden, dim=1)
        T_gate_level_one = torch.sigmoid(t_g_level_one)
        S_gate_level_one = torch.sigmoid(s_g_level_one)
        T_new_level_one = T_gate_level_one * t_t_level_one + (1 - T_gate_level_one) * s_t_level_one
        S_new_level_one = S_gate_level_one * s_s_level_one + (1 - S_gate_level_one) * t_s_level_one


        t_att_gate_level_two = torch.sigmoid(t_next_level_two)
        T_fusion_level_two = T_t_level_two * t_att_gate_level_two + (1 - t_att_gate_level_two) * T_trend_level_two
        T_concat_level_two = self.conv_t_level_two(T_fusion_level_two)
        S_concat_level_two = self.conv_s_level_two(S_t_level_two)
        t_g_level_two, t_t_level_two, t_s_level_two = torch.split(T_concat_level_two, self.num_hidden, dim=1)
        s_g_level_two, s_t_level_two, s_s_level_two = torch.split(S_concat_level_two, self.num_hidden, dim=1)
        T_gate_level_two = torch.sigmoid(t_g_level_two)
        S_gate_level_two = torch.sigmoid(s_g_level_two)
        T_new_level_two = T_gate_level_two * t_t_level_two + (1 - T_gate_level_two) * s_t_level_two
        S_new_level_two = S_gate_level_two * s_s_level_two + (1 - S_gate_level_two) * t_s_level_two


        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend
        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s

        # version 1
        # T_new_return = T_new + T_new_level_one + T_new_level_two
        # S_new_return = S_new + S_new_level_one + S_new_level_two

        # T_new_level_one_return = T_new + T_new_level_one + T_new_level_two
        # S_new_level_one_return = S_new + S_new_level_one + S_new_level_two

        # T_new_level_two_return = T_new + T_new_level_two + T_new_level_one
        # S_new_level_two_return = S_new + S_new_level_two + S_new_level_one

        T_new_gate = torch.sigmoid(T_new)
        T_new_1 =  T_new_gate * T_new + (1 - T_new_gate) * T_new_level_one
        T_new_1_gate = torch.sigmoid(T_new_1)
        T_new_2 = T_new_1_gate * T_new_1 + (1 - T_new_1_gate) * T_new_level_two

        S_new_gate = torch.sigmoid(S_new)
        S_new_1 = S_new_gate * S_new + (1 - S_new_gate) * S_new_level_one
        S_new_1_gate = torch.sigmoid(S_new_1)
        S_new_2 = S_new_1_gate * S_new_1 + (1 - S_new_1_gate) * S_new_level_two

        T_new_level_one_gate = torch.sigmoid(T_new_level_one)
        T_new_level_one_1 = T_new_level_one_gate * T_new_level_one + (1 - T_new_level_one_gate) * T_new
        T_new_level_one_1_gate = torch.sigmoid(T_new_level_one_1)
        T_new_level_one_2 = T_new_level_one_1_gate * T_new_level_one_1 + (1 - T_new_level_one_1_gate) * T_new_level_two

        S_new_level_one_gate = torch.sigmoid(S_new_level_one)
        S_new_level_one_1 = S_new_level_one_gate * S_new_level_one + (1 - S_new_level_one_gate) * S_new
        S_new_level_one_1_gate = torch.sigmoid(S_new_level_one_1)
        S_new_level_one_2 = S_new_level_one_1_gate * S_new_level_one_1 + (1 - S_new_level_one_1_gate) * S_new_level_two

        T_new_level_two_gate = torch.sigmoid(T_new_level_two)
        T_new_level_two_1 = T_new_level_two_gate * T_new_level_two + (1 - T_new_level_two_gate) * T_new
        T_new_level_two_1_gate = torch.sigmoid(T_new_level_two_1)
        T_new_level_two_2 = T_new_level_two_1_gate * T_new_level_two_1 + (1 - T_new_level_two_1_gate) * T_new_level_one

        S_new_level_two_gate = torch.sigmoid(S_new_level_two)
        S_new_level_two_1 = S_new_level_two_gate * S_new_level_two + (1 - S_new_level_two_gate) * S_new
        S_new_level_two_1_gate = torch.sigmoid(S_new_level_two_1)
        S_new_level_two_2 = S_new_level_two_1_gate * S_new_level_two_1 + (1 - S_new_level_two_1_gate) * S_new_level_one

        # version 2
        # T_new = 0.5 * T_new + 0.3 * T_concat_level_one + 0.2 * T_concat_level_two
        # S_new = 0.5 * S_new + 0.3 * S_new_level_one + 0.2 * S_new_level_two

       # version3
       # iAFF AFF

        # x,residual  [B,C,H,W]
        # T_new_level_two_one = self.attention_t1(T_new_level_two, T_new_level_one)
        # T_new = self.attention_t2(T_new_level_two_one, T_new)
        # S_new_level_two_one = self.attention_s1(S_new_level_two, S_new_level_one)
        # S_new = self.attention_s2(S_new_level_two_one, S_new)


        if self.cell_mode == 'residual':
            S_new_2 = S_new_2 + S_t
        return T_new_2, T_new_level_one_2, T_new_level_two_2, S_new_2, S_new_level_one_2, S_new_level_two_2
