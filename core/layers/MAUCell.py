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

        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights_list_s = []
        weights_list_t = []
        for i in range(self.tau):
            weights_list_s.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        for i in range(self.tau):
            weights_list_t.append((t_att[i] * t_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_s = torch.stack(weights_list_s, dim=0)
        weights_list_s = torch.reshape(weights_list_s, (*weights_list_s.shape, 1, 1, 1))
        weights_list_s = self.softmax(weights_list_s)

        weights_list_t = torch.stack(weights_list_t, dim=0)
        weights_list_t = torch.reshape(weights_list_t, (*weights_list_t.shape, 1, 1, 1))
        weights_list_t = self.softmax(weights_list_t)

        T_trend = t_att * weights_list_s
        T_trend = T_trend.sum(dim=0)
        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend
        T_concat = self.conv_t(T_fusion)

        S_trend = s_att * weights_list_t
        S_trend = S_trend.sum(dim=0)
        s_att_gate = torch.sigmoid(s_next)
        S_fusion = S_t * s_att_gate + (1 - s_att_gate) * S_trend
        S_concat = self.conv_s(S_fusion)

        #S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s

        ###################################################################

        s_next_2 = self.conv_s_next_2(S_t)
        t_next_2 = self.conv_t_next_2(T_t)
        weights_list_s_2 = []
        weights_list_t_2 = []
        for i in range(self.tau):
            weights_list_s_2.append((s_att[i] * s_next_2).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        for i in range(self.tau):
            weights_list_t_2.append((t_att[i] * t_next_2).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_s_2 = torch.stack(weights_list_s_2, dim=0)
        weights_list_s_2 = torch.reshape(weights_list_s_2, (*weights_list_s_2.shape, 1, 1, 1))
        weights_list_s_2 = self.softmax(weights_list_s_2)

        weights_list_t_2 = torch.stack(weights_list_t_2, dim=0)
        weights_list_t_2 = torch.reshape(weights_list_t_2, (*weights_list_t_2.shape, 1, 1, 1))
        weights_list_t_2 = self.softmax(weights_list_t_2)

        T_trend_2 = t_att * weights_list_s_2
        T_trend_2 = T_trend_2.sum(dim=0)
        t_att_gate_2 = torch.sigmoid(t_next_2)
        T_fusion_2 = T_t * t_att_gate_2 + (1 - t_att_gate_2) * T_trend_2
        T_concat_2 = self.conv_t_2(T_fusion_2)

        S_trend_2 = s_att * weights_list_t_2
        S_trend_2 = S_trend_2.sum(dim=0)
        s_att_gate_2 = torch.sigmoid(s_next_2)
        S_fusion_2 = S_t * s_att_gate_2 + (1 - s_att_gate_2) * S_trend_2
        S_concat_2 = self.conv_s_2(S_fusion_2)

        #S_concat_2 = self.conv_s_2(S_t)
        t_g_2, t_t_2, t_s_2 = torch.split(T_concat_2, self.num_hidden, dim=1)
        s_g_2, s_t_2, s_s_2 = torch.split(S_concat_2, self.num_hidden, dim=1)
        T_gate_2 = torch.sigmoid(t_g_2)
        S_gate_2 = torch.sigmoid(s_g_2)
        T_new_2 = T_gate_2 * t_t_2 + (1 - T_gate_2) * s_t_2
        S_new_2 = S_gate_2 * s_s_2 + (1 - S_gate_2) * t_s_2

        ###################################################################

        s_next_3 = self.conv_s_next_3(S_t)
        t_next_3 = self.conv_t_next_3(T_t)
        weights_list_s_3 = []
        weights_list_t_3 = []
        for i in range(self.tau):
            weights_list_s_3.append((s_att[i] * s_next_3).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        for i in range(self.tau):
            weights_list_t_3.append((t_att[i] * t_next_3).sum(dim=(1, 2, 3)) / math.sqrt(self.d))

        weights_list_s_3 = torch.stack(weights_list_s_3, dim=0)
        weights_list_s_3 = torch.reshape(weights_list_s_3, (*weights_list_s_3.shape, 1, 1, 1))
        weights_list_s_3 = self.softmax(weights_list_s_3)

        weights_list_t_3 = torch.stack(weights_list_t_3, dim=0)
        weights_list_t_3 = torch.reshape(weights_list_t_3, (*weights_list_t_3.shape, 1, 1, 1))
        weights_list_t_3 = self.softmax(weights_list_t_3)

        T_trend_3 = t_att * weights_list_s_3
        T_trend_3 = T_trend_3.sum(dim=0)
        t_att_gate_3 = torch.sigmoid(t_next_3)
        T_fusion_3 = T_t * t_att_gate_3 + (1 - t_att_gate_3) * T_trend_3
        T_concat_3 = self.conv_t_3(T_fusion_3)

        S_trend_3 = s_att * weights_list_t_3
        S_trend_3 = S_trend_3.sum(dim=0)
        s_att_gate_3 = torch.sigmoid(s_next_3)
        S_fusion_3 = S_t * s_att_gate_3 + (1 - s_att_gate_3) * S_trend_3
        S_concat_3 = self.conv_s_3(S_fusion_3)

        #S_concat_3 = self.conv_s_3(S_t)
        t_g_3, t_t_3, t_s_3 = torch.split(T_concat_3, self.num_hidden, dim=1)
        s_g_3, s_t_3, s_s_3 = torch.split(S_concat_3, self.num_hidden, dim=1)
        T_gate_3 = torch.sigmoid(t_g_3)
        S_gate_3 = torch.sigmoid(s_g_3)
        T_new_3 = T_gate_3 * t_t_3 + (1 - T_gate_3) * s_t_3
        S_new_3 = S_gate_3 * s_s_3 + (1 - S_gate_3) * t_s_3


        # if self.cell_mode == 'residual':
        #     S_new = S_new + S_t
        #T_new_gate = torch.sigmoid(T_new)
        T_new_2_gate = torch.sigmoid(T_new_2)
        T_new_3_gate = torch.sigmoid(T_new_3)

        T_new_2 = T_new_2 * T_new_2_gate + T_new * (1 - T_new_2_gate)
        T_new_return = T_new_3 * T_new_3_gate + T_new_2 * (1 - T_new_3_gate)

        #T_new_return = T_new_3

        #S_new_gate = torch.sigmoid(S_new)
        S_new_2_gate = torch.sigmoid(S_new_2)
        S_new_3_gate = torch.sigmoid(S_new_3)

        S_new_2 = S_new_2 * S_new_2_gate + S_new * (1 - S_new_2_gate)
        S_new_return = S_new_3 * S_new_3_gate + S_new_2 * (1 - S_new_3_gate)

        #S_new_return =S_new_3

        return T_new_return, S_new_return
