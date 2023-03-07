import torch
import torch.nn as nn
import math


class MAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=7, stride=1, padding=3,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=7, stride=1, padding=3,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=7, stride=1, padding=3,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=7, stride=1, padding=3,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_t_2 = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next_2 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s_2 = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next_2 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_t_3 = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=3, stride=1, padding=1,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next_3 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=3, stride=1, padding=1,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s_3 = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=3, stride=1, padding=1,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next_3 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_t_lower_t_1 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_lower_t_2 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_lower_s_1 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_lower_s_2 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_t_upper_t_1 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_upper_t_2 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_upper_s_1 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_upper_s_2 = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1, padding=2,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)
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

        s_next_2 = self.conv_s_next_2(S_t)
        t_next_2 = self.conv_t_next_2(T_t)
        weights_list_2 = []
        for i in range(self.tau):
            weights_list_2.append((s_att[i] * s_next_2).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_2 = torch.stack(weights_list_2, dim=0)
        weights_list_2 = torch.reshape(weights_list_2, (*weights_list_2.shape, 1, 1, 1))
        weights_list_2 = self.softmax(weights_list_2)
        T_trend_2 = t_att * weights_list_2
        T_trend_2 = T_trend_2.sum(dim=0)
        t_att_gate_2 = torch.sigmoid(t_next_2)
        T_fusion_2 = T_t * t_att_gate_2 + (1 - t_att_gate_2) * T_trend_2
        T_concat_2 = self.conv_t_2(T_fusion_2)
        S_concat_2 = self.conv_s_2(S_t)
        t_g_2, t_t_2, t_s_2 = torch.split(T_concat_2, self.num_hidden, dim=1)
        s_g_2, s_t_2, s_s_2 = torch.split(S_concat_2, self.num_hidden, dim=1)
        T_gate_2 = torch.sigmoid(t_g_2)
        S_gate_2 = torch.sigmoid(s_g_2)
        T_new_2 = T_gate_2 * t_t_2 + (1 - T_gate_2) * s_t_2
        S_new_2 = S_gate_2 * s_s_2 + (1 - S_gate_2) * t_s_2

        s_next_3 = self.conv_s_next_3(S_t)
        t_next_3 = self.conv_t_next_3(T_t)
        weights_list_3 = []
        for i in range(self.tau):
            weights_list_3.append((s_att[i] * s_next_3).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_3 = torch.stack(weights_list_3, dim=0)
        weights_list_3 = torch.reshape(weights_list_3, (*weights_list_3.shape, 1, 1, 1))
        weights_list_3 = self.softmax(weights_list_3)
        T_trend_3 = t_att * weights_list_3
        T_trend_3 = T_trend_3.sum(dim=0)
        t_att_gate_3 = torch.sigmoid(t_next_3)
        T_fusion_3 = T_t * t_att_gate_3 + (1 - t_att_gate_3) * T_trend_3
        T_concat_3 = self.conv_t_3(T_fusion_3)
        S_concat_3 = self.conv_s_3(S_t)
        t_g_3, t_t_3, t_s_3 = torch.split(T_concat_3, self.num_hidden, dim=1)
        s_g_3, s_t_3, s_s_3 = torch.split(S_concat_3, self.num_hidden, dim=1)
        T_gate_3 = torch.sigmoid(t_g_3)
        S_gate_3 = torch.sigmoid(s_g_3)
        T_new_3 = T_gate_3 * t_t_3 + (1 - T_gate_3) * s_t_3
        S_new_3 = S_gate_3 * s_s_3 + (1 - S_gate_3) * t_s_3

        # if self.cell_mode == 'residual':
        #     S_new = S_new + S_t
        # T_new_gate = torch.sigmoid(T_new)
        # T_new_2_gate = torch.sigmoid(T_new_2)
        # T_new_3_gate = torch.sigmoid(T_new_3)

        T_new_3_upper = self.conv_t_upper_t_1(T_new_3)
        T_new_2_gate = torch.sigmoid(T_new_2)
        T_new_2 = T_new_2 * T_new_2_gate + (1 - T_new_2_gate) * T_new_3_upper
        T_new_2_upper = self.conv_t_upper_t_2(T_new_2)
        T_new_gate = torch.sigmoid(T_new)
        T_new = T_new * T_new_gate + (1 - T_new_gate) * T_new_2_upper

        T_new_lower = self.conv_t_lower_t_1(T_new)
        T_new_2_gate = torch.sigmoid(T_new_2)
        T_new_2 = T_new_2 * T_new_2_gate + (1 - T_new_2_gate) * T_new_lower
        T_new_2_lower = self.conv_t_lower_t_2(T_new_2)
        T_new_3_gate = torch.sigmoid(T_new_3)
        T_new_return = T_new_3 * T_new_3_gate + (1 - T_new_3_gate) * T_new_2_lower

        # T_new_return = T_new_3

        # S_new_gate = torch.sigmoid(S_new)
        # S_new_2_gate = torch.sigmoid(S_new_2)
        # S_new_3_gate = torch.sigmoid(S_new_3)
        #
        # S_new_2 = S_new_2 * S_new_2_gate + S_new * (1 - S_new_2_gate)
        # S_new_return = S_new_3 * S_new_3_gate + S_new_2 * (1 - S_new_3_gate)

        S_new_3_upper = self.conv_s_upper_s_1(S_new_3)
        S_new_2_gate = torch.sigmoid(S_new_2)
        S_new_2 = S_new_2 * S_new_2_gate + (1 - S_new_2_gate) * S_new_3_upper
        S_new_2_upper = self.conv_s_upper_s_2(S_new_2)
        S_new_gate = torch.sigmoid(S_new)
        S_new = S_new * S_new_gate + (1 - S_new_gate) * S_new_2_upper

        S_new_lower = self.conv_s_lower_s_1(S_new)
        S_new_2_gate = torch.sigmoid(S_new_2)
        S_new_2 = S_new_2 * S_new_2_gate + (1 - S_new_2_gate) * S_new_lower
        S_new_2_lower = self.conv_s_lower_s_2(S_new_2)
        S_new_3_gate = torch.sigmoid(S_new_3)
        S_new_return = T_new_3 * S_new_3_gate + (1 - S_new_3_gate) * S_new_2_lower

        # S_new_return =S_new_3

        return T_new_return, S_new_return
