import torch
import torch.nn as nn
import math


class MAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size_one, filter_size_two, filter_size_three, filter_size_four,
                 stride, tau_one, tau_two, tau_three,tau_four, cell_mode):
        super(MAUCell, self).__init__()
        # in_channel = 1 ,num_hidden[i] = 64,  height = 16 , width = 16,
        # filter_size = (5,5), stride = 1,tau = 5,cell_mode = normal
        # num_hidden = 64
        self.num_hidden = num_hidden
        # padding = (2,2)
        self.padding_one = (3, 3)
        self.padding_two = (2, 2)
        self.padding_three = (1, 1)
        self.padding_four = (0, 0)
        self.cell_mode = cell_mode
        # d = 64 * 16 * 16 = 16384
        self.d = num_hidden * height * width
        # tau = 5
        self.tau_one = tau_one
        self.tau_two = tau_two
        self.tau_three = tau_three
        self.tau_four = tau_four
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t_level_one = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_one, stride=stride, padding=self.padding_one,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_level_two = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_two, stride=stride, padding=self.padding_two,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_level_three = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_three, stride=stride, padding=self.padding_three,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_level_four = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_four, stride=stride,padding=self.padding_four,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_1 = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_one, stride=stride, padding=self.padding_one,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_2 = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_two, stride=stride, padding=self.padding_two,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_3 = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_three, stride=stride, padding=self.padding_three,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_t_4 = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size_four, stride=stride, padding=self.padding_four,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_t_next_level_one = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size_one, stride=stride, padding=self.padding_one,),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_next_level_two = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size_two, stride=stride, padding=self.padding_two,),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_t_next_level_three = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size_three, stride=stride, padding=self.padding_three,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_t_next_level_four = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size_four, stride=stride, padding=self.padding_four,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s_level_one = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_one, stride=stride, padding=self.padding_one,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_level_two = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_two, stride=stride, padding=self.padding_two,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_level_three = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_three, stride=stride, padding=self.padding_three,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_level_four = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_four, stride=stride, padding=self.padding_four,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_1 = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_one, stride=stride, padding=self.padding_one,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_2 = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_two, stride=stride, padding=self.padding_two,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_3 = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_three, stride=stride, padding=self.padding_three,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_4 = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size_four, stride=stride,padding=self.padding_four,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_next_level_one = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size_one, stride=stride, padding=self.padding_one,),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_s_next_level_two = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size_two, stride=stride, padding=self.padding_two,),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_s_next_level_three = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size_three, stride=stride, padding=self.padding_three,),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s_next_level_four = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size_four, stride=stride, padding=self.padding_four,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

        # self.attention_s1 = AFF(channels=64)
        # self.attention_s2 = AFF(channels=64)
        # self.attention_t1 = AFF(channels=64)
        # self.attention_t2 = AFF(channels=64)

    def forward(self, T_t_level_one, T_t_level_two, T_t_level_three, T_t_level_four, S_t_level_one, S_t_level_two, S_t_level_three, S_t_level_four,
                t_att_level_one, s_att_level_one, t_att_level_two, s_att_level_two, t_att_level_three, s_att_level_three, t_att_level_four, s_att_level_four):
        s_next_level_one = self.conv_s_next_level_one(S_t_level_one)
        t_next_level_one = self.conv_t_next_level_one(T_t_level_one)

        s_next_level_two = self.conv_s_next_level_two(S_t_level_two)
        t_next_level_two = self.conv_t_next_level_two(T_t_level_two)

        s_next_level_three = self.conv_s_next_level_three(S_t_level_three)
        t_next_level_three = self.conv_t_next_level_three(T_t_level_three)

        s_next_level_four = self.conv_s_next_level_four(S_t_level_four)
        t_next_level_four = self.conv_t_next_level_four(T_t_level_four)


        weights_list_level_one = []
        for i in range(self.tau_one):
            weights_list_level_one.append((s_att_level_one[i] * s_next_level_one).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_level_one = torch.stack(weights_list_level_one, dim=0)
        weights_list_level_one = torch.reshape(weights_list_level_one, (*weights_list_level_one.shape, 1, 1, 1))
        weights_list_level_one = self.softmax(weights_list_level_one)
        T_trend_level_one = t_att_level_one * weights_list_level_one
        T_trend_level_one = T_trend_level_one.sum(dim=0)

        weights_list_level_two = []
        for i in range(self.tau_two):
            weights_list_level_two.append((s_att_level_two[i] * s_next_level_two).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_level_two = torch.stack(weights_list_level_two, dim=0)
        weights_list_level_two = torch.reshape(weights_list_level_two, (*weights_list_level_two.shape, 1, 1, 1))
        weights_list_level_two = self.softmax(weights_list_level_two)
        T_trend_level_two = t_att_level_two * weights_list_level_two
        T_trend_level_two = T_trend_level_two.sum(dim=0)


        weights_list_level_three = []
        for i in range(self.tau_three):
            weights_list_level_three.append((s_att_level_three[i] * s_next_level_three).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_level_three = torch.stack(weights_list_level_three, dim=0)
        weights_list_level_three = torch.reshape(weights_list_level_three, (*weights_list_level_three.shape, 1, 1, 1))
        weights_list_level_three = self.softmax(weights_list_level_three)
        T_trend_level_three = t_att_level_three * weights_list_level_three
        T_trend_level_three = T_trend_level_three.sum(dim=0)

        weights_list_level_four = []
        for i in range(self.tau_four):
            weights_list_level_four.append((s_att_level_four[i] * s_next_level_four).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list_level_four = torch.stack(weights_list_level_four, dim=0)
        weights_list_level_four = torch.reshape(weights_list_level_four, (*weights_list_level_four.shape, 1, 1, 1))
        weights_list_level_four = self.softmax(weights_list_level_four)
        T_trend_level_four = t_att_level_four * weights_list_level_four
        T_trend_level_four = T_trend_level_four.sum(dim=0)


        t_att_gate_level_one = torch.sigmoid(t_next_level_one)
        T_fusion_level_one = T_t_level_one * t_att_gate_level_one + (1 - t_att_gate_level_one) * T_trend_level_one
        T_concat_level_one = self.conv_t_level_one(T_fusion_level_one)
        S_concat_levle_one = self.conv_s_level_one(S_t_level_one)
        t_g_level_one, t_t_level_one, t_s_level_one = torch.split(T_concat_level_one, self.num_hidden, dim=1)
        s_g_level_one, s_t_level_one, s_s_level_one = torch.split(S_concat_levle_one, self.num_hidden, dim=1)
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


        t_att_gate_level_three = torch.sigmoid(t_next_level_three)
        T_fusion_level_three = T_t_level_three * t_att_gate_level_three + (1 - t_att_gate_level_three) * T_trend_level_three
        T_concat_level_three = self.conv_t_level_three(T_fusion_level_three)
        S_concat_level_three = self.conv_s_level_three(S_t_level_three)
        t_g_level_three, t_t_level_three, t_s_level_three = torch.split(T_concat_level_three, self.num_hidden, dim=1)
        s_g_level_three, s_t_level_three, s_s_level_three = torch.split(S_concat_level_three, self.num_hidden, dim=1)
        T_gate_level_three = torch.sigmoid(t_g_level_three)
        S_gate_level_three = torch.sigmoid(s_g_level_three)
        T_new_level_three = T_gate_level_three * t_t_level_three + (1 - T_gate_level_three) * s_t_level_three
        S_new_level_three = S_gate_level_three * s_s_level_three + (1 - S_gate_level_three) * t_s_level_three


        t_att_gate_level_four = torch.sigmoid(t_next_level_four)
        T_fusion_level_four = T_t_level_four * t_att_gate_level_four + (1 - t_att_gate_level_four) * T_trend_level_four
        T_concat_level_four = self.conv_t_level_four(T_fusion_level_four)
        S_concat_level_four = self.conv_s_level_four(S_t_level_four)
        t_g_level_four, t_t_level_four, t_s_level_four = torch.split(T_concat_level_four, self.num_hidden, dim=1)
        s_g_level_four, s_t_level_four, s_s_level_four = torch.split(S_concat_level_four, self.num_hidden, dim=1)
        T_gate_level_four = torch.sigmoid(t_g_level_four)
        S_gate_level_four = torch.sigmoid(s_g_level_four)
        T_new_level_four = T_gate_level_four * t_t_level_four + (1 - T_gate_level_four) * s_t_level_four
        S_new_level_four = S_gate_level_four * s_s_level_four + (1 - S_gate_level_four) * t_s_level_four




        T_new_level_one_concat = self.conv_t_1(T_new_level_one)
        S_new_level_one_concat = self.conv_s_1(S_new_level_one)

        T_new_level_two_concat = self.conv_t_2(T_new_level_two)
        S_new_level_two_concat = self.conv_s_2(S_new_level_two)

        T_new_level_three_concat = self.conv_t_3(T_new_level_three)
        S_new_level_three_concat = self.conv_s_3(S_new_level_three)

        T_new_level_four_concat = self.conv_t_4(T_new_level_four)
        S_new_level_four_concat = self.conv_s_4(S_new_level_four)

        t_g_one, t_t_one, t_s_one = torch.split(T_new_level_one_concat, self.num_hidden, dim=1)
        s_g_one, s_t_one, s_s_one = torch.split(S_new_level_one_concat, self.num_hidden, dim=1)

        t_g_two, t_t_two, t_s_two = torch.split(T_new_level_two_concat, self.num_hidden, dim=1)
        s_g_two, s_t_two, s_s_two = torch.split(S_new_level_two_concat, self.num_hidden, dim=1)

        t_g_three, t_t_three, t_s_three = torch.split(T_new_level_three_concat, self.num_hidden, dim=1)
        s_g_three, s_t_three, s_s_three = torch.split(S_new_level_three_concat, self.num_hidden, dim=1)

        t_g_four, t_t_four, t_s_four = torch.split(T_new_level_four_concat, self.num_hidden, dim=1)
        s_g_four, s_t_four, s_s_four = torch.split(S_new_level_four_concat, self.num_hidden, dim=1)

        T_gate_one = torch.sigmoid(t_g_one)
        S_gate_one = torch.sigmoid(s_g_one)

        T_gate_two = torch.sigmoid(t_g_two)
        S_gate_two = torch.sigmoid(s_g_two)

        T_gate_three = torch.sigmoid(t_g_three)
        S_gate_three = torch.sigmoid(s_g_three)

        T_gate_four = torch.sigmoid(t_g_four)
        S_gate_four = torch.sigmoid(s_g_four)

        T_new_level_one_return = T_gate_three * (T_gate_two * (T_gate_one * t_t_one + (1 - T_gate_one) * t_t_two) + (1 - T_gate_two) * t_t_three) + (1 - T_gate_three) * t_t_four
        S_new_level_one_return = S_gate_three * (S_gate_two * (S_gate_one * s_t_one + (1 - S_gate_one) * s_t_two) + (1 - S_gate_two) * s_t_three) + (1 - S_gate_three) * s_t_four

        T_new_level_two_return = T_gate_three * (T_gate_one * (T_gate_two * t_t_two + (1 - T_gate_two) * t_t_one) + (1 - T_gate_one) * t_t_three) + (1 - T_gate_three) * t_t_four
        S_new_level_two_return = S_gate_three * (S_gate_one * (S_gate_two * s_t_two + (1 - S_gate_two) * s_t_one) + (1 - S_gate_one) * s_t_three) + (1 - S_gate_three) * s_t_four

        T_new_level_three_return = T_gate_two * (T_gate_four * (T_gate_three * t_t_three + (1 - T_gate_three) * t_t_four) + (1 - T_gate_four) * t_t_two) + (1 - T_gate_two) * t_t_one
        S_new_level_three_return = S_gate_two * (S_gate_four * (S_gate_three * s_t_three + (1 - S_gate_three) * s_t_four) + (1 - S_gate_four) * s_t_two) + (1 - T_gate_two) * t_t_one

        T_new_level_four_return = T_gate_two * (T_gate_three * (T_gate_four * t_t_four + (1 - T_gate_four) * t_t_three) + (1 - T_gate_three) * t_t_two) + (1 - T_gate_two) * t_t_one
        S_new_level_four_return =  S_gate_two * (S_gate_three * (S_gate_four * s_t_four + (1 - S_gate_four) * s_t_three) + (1 - S_gate_three) * s_t_two) + (1 - T_gate_two) * t_t_one

        if self.cell_mode == 'residual':
            S_new_level_one_return = S_new_level_one_return + S_t_level_one
        return T_new_level_one_return, T_new_level_two_return, T_new_level_three_return, T_new_level_four_return, S_new_level_one_return, S_new_level_two_return, S_new_level_three_return,S_new_level_four_return
