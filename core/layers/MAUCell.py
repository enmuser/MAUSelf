import torch
import torch.nn as nn
import math


class MAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode, num_layers):
        super(MAUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.num_layers = num_layers
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_next_pre = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_next_pre_cat = nn.Sequential(
            nn.Conv2d(6 * num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s_next_pre = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s_next_pre_cat = nn.Sequential(
            nn.Conv2d(6 * num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att, index, T_pre, S_pre):
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

        T_pre_history = None
        S_pre_history = None
        for curIndex in reversed(range(self.num_layers)):
            if curIndex != index:
                T_pre_history_tmp = T_pre[curIndex][-self.tau:]
                T_pre_history_tmp = torch.stack(T_pre_history_tmp, dim=0)
                T_pre_history_tmp = T_pre_history_tmp.sum(dim=0)
                T_pre_history_tmp = self.conv_t_next_pre(T_pre_history_tmp)
                if T_pre_history is None:
                    T_pre_history = T_pre_history_tmp
                else:
                    T_pre_history = torch.cat([T_pre_history, T_pre_history_tmp], 1)
                S_pre_history_tmp = S_pre[curIndex][-self.tau:]
                S_pre_history_tmp = torch.stack(S_pre_history_tmp, dim=0)
                S_pre_history_tmp = S_pre_history_tmp.sum(dim=0)
                S_pre_history_tmp = self.conv_s_next_pre(S_pre_history_tmp)
                if S_pre_history is None:
                    S_pre_history = S_pre_history_tmp
                else:
                    S_pre_history = torch.cat([S_pre_history, S_pre_history_tmp], 1)

        T_concat = self.conv_t(T_fusion)
        T_concat = torch.cat([T_concat, T_pre_history], 1)
        T_concat = self.conv_t_next_pre_cat(T_concat)

        S_concat = self.conv_s(S_t)
        S_concat = torch.cat([S_concat, S_pre_history], 1)
        S_concat = self.conv_s_next_pre_cat(S_concat)

        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        if self.cell_mode == 'residual':
            S_new = S_new + S_t
        return T_new, S_new
