import torch
import torch.nn as nn
from core.layers.MAUCell import MAUCell
import math
import random

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau_one = configs.tau_one
        self.tau_two = configs.tau_two
        self.tau_three = configs.tau_three
        self.tau_four = configs.tau_four
        self.cell_mode = configs.cell_mode
        self.train_level_base_line = configs.train_level_base_line
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []

        width = configs.img_width // configs.patch_size // configs.sr_size
        height = configs.img_height // configs.patch_size // configs.sr_size

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size_one, configs.filter_size_two,
                        configs.filter_size_three, configs.filter_size_four, configs.stride, self.tau_one, self.tau_two, self.tau_three,self.tau_four,self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders_level_one = []
        encoder_level_one = nn.Sequential()
        encoder_level_one.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder_level_one.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders_level_one.append(encoder_level_one)
        for i in range(n):
            encoder_level_one = nn.Sequential()
            encoder_level_one.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder_level_one.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders_level_one.append(encoder_level_one)
        self.encoders_level_one = nn.ModuleList(encoders_level_one)

        encoders_level_two = []
        encoder_level_two = nn.Sequential()
        encoder_level_two.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder_level_two.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders_level_two.append(encoder_level_two)
        for i in range(n):
            encoder_level_two = nn.Sequential()
            encoder_level_two.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder_level_two.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders_level_two.append(encoder_level_two)
        self.encoders_level_two = nn.ModuleList(encoders_level_two)


        encoders_level_three = []
        encoder_level_three = nn.Sequential()
        encoder_level_three.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder_level_three.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders_level_three.append(encoder_level_three)
        for i in range(n):
            encoder_level_three = nn.Sequential()
            encoder_level_three.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder_level_three.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders_level_three.append(encoder_level_three)
        self.encoders_level_three = nn.ModuleList(encoders_level_three)

        encoders_level_four = []
        encoder_level_four = nn.Sequential()
        encoder_level_four.add_module(name='encoder_t_conv{0}'.format(-1),
                                       module=nn.Conv2d(in_channels=self.frame_channel,
                                                        out_channels=self.num_hidden[0],
                                                        stride=1,
                                                        padding=0,
                                                        kernel_size=1))
        encoder_level_four.add_module(name='relu_t_{0}'.format(-1),
                                       module=nn.LeakyReLU(0.2))
        encoders_level_four.append(encoder_level_four)
        for i in range(n):
            encoder_level_four = nn.Sequential()
            encoder_level_four.add_module(name='encoder_t{0}'.format(i),
                                           module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                            out_channels=self.num_hidden[0],
                                                            stride=(2, 2),
                                                            padding=(1, 1),
                                                            kernel_size=(3, 3)
                                                            ))
            encoder_level_four.add_module(name='encoder_t_relu{0}'.format(i),
                                           module=nn.LeakyReLU(0.2))
            encoders_level_four.append(encoder_level_four)
        self.encoders_level_four = nn.ModuleList(encoders_level_four)

        # Decoder
        decoders_level_one = []

        for i in range(n - 1):
            decoder_level_one = nn.Sequential()
            decoder_level_one.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder_level_one.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders_level_one.append(decoder_level_one)

        if n > 0:
            decoder_level_one = nn.Sequential()
            decoder_level_one.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders_level_one.append(decoder_level_one)
        self.decoders_level_one = nn.ModuleList(decoders_level_one)

        decoders_level_two = []
        for i in range(n - 1):
            decoder_level_two = nn.Sequential()
            decoder_level_two.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder_level_two.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders_level_two.append(decoder_level_two)

        if n > 0:
            decoder_level_two = nn.Sequential()
            decoder_level_two.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders_level_two.append(decoder_level_two)
        self.decoders_level_two = nn.ModuleList(decoders_level_two)

        decoders_level_three = []
        for i in range(n - 1):
            decoder_level_three = nn.Sequential()
            decoder_level_three.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder_level_three.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders_level_three.append(decoder_level_three)

        if n > 0:
            decoder_level_three = nn.Sequential()
            decoder_level_three.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders_level_three.append(decoder_level_three)
        self.decoders_level_three = nn.ModuleList(decoders_level_three)

        decoders_level_four = []
        for i in range(n - 1):
            decoder_level_four = nn.Sequential()
            decoder_level_four.add_module(name='c_decoder{0}'.format(i),
                                           module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                     out_channels=self.num_hidden[-1],
                                                                     stride=(2, 2),
                                                                     padding=(1, 1),
                                                                     kernel_size=(3, 3),
                                                                     output_padding=(1, 1)
                                                                     ))
            decoder_level_four.add_module(name='c_decoder_relu{0}'.format(i),
                                           module=nn.LeakyReLU(0.2))
            decoders_level_four.append(decoder_level_four)

        if n > 0:
            decoder_level_four = nn.Sequential()
            decoder_level_four.add_module(name='c_decoder{0}'.format(n - 1),
                                           module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                                     out_channels=self.num_hidden[-1],
                                                                     stride=(2, 2),
                                                                     padding=(1, 1),
                                                                     kernel_size=(3, 3),
                                                                     output_padding=(1, 1)
                                                                     ))
            decoders_level_four.append(decoder_level_four)
        self.decoders_level_four = nn.ModuleList(decoders_level_four)

        self.srcnn_level_one = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.srcnn_level_two = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.srcnn_level_three = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.srcnn_level_four = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames_level_one, frames_level_two, frames_level_three,
                mask_tensor_one,mask_tensor_two,mask_tensor_three,mask_tensor_four, itr):
        # print('ok')
        mask_true_one = mask_tensor_one.permute(0, 1, 4, 2, 3).contiguous()
        mask_true_two = mask_tensor_two.permute(0, 1, 4, 2, 3).contiguous()
        mask_true_three = mask_tensor_three.permute(0, 1, 4, 2, 3).contiguous()
        mask_true_four = mask_tensor_four.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames_level_one.shape[0]
        height = frames_level_one.shape[3] // self.configs.sr_size
        width = frames_level_one.shape[4] // self.configs.sr_size
        frame_channels = frames_level_one.shape[2]
        next_frames = []
        T_t_level_one = []
        T_t_level_two = []
        T_t_level_three = []
        T_t_level_four = []
        T_pre_level_one = []
        S_pre_level_one = []
        T_pre_level_two = []
        S_pre_level_two = []
        T_pre_level_three = []
        S_pre_level_three = []
        T_pre_level_four = []
        S_pre_level_four = []
        x_gen_level_one = None
        for layer_idx in range(self.num_layers):
            tmp_t_level_one = []
            tmp_s_level_one = []
            tmp_t_level_two = []
            tmp_s_level_two = []
            tmp_t_level_three = []
            tmp_s_level_three = []
            tmp_t_level_four = []
            tmp_s_level_four = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau_one):
                tmp_t_level_one.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s_level_one.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            for i in range(self.tau_two):
                tmp_t_level_two.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s_level_two.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            for i in range(self.tau_three):
                tmp_t_level_three.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s_level_three.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            for i in range(self.tau_four):
                tmp_t_level_four.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s_level_four.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre_level_one.append(tmp_t_level_one)
            S_pre_level_one.append(tmp_s_level_one)
            T_pre_level_two.append(tmp_t_level_two)
            S_pre_level_two.append(tmp_s_level_two)
            T_pre_level_three.append(tmp_t_level_three)
            S_pre_level_three.append(tmp_s_level_three)
            T_pre_level_four.append(tmp_t_level_four)
            S_pre_level_four.append(tmp_s_level_four)


        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net_level_one = frames_level_one[:, t]
                net_level_two = frames_level_one[:, t]
                net_level_three = frames_level_one[:, t]
                net_level_four = frames_level_one[:, t]
            else:
                time_diff = t - self.configs.input_length
                #net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net_mask = mask_true[:, time_diff] * frames_mask[:, t] + (1 - mask_true[:, time_diff]) * x_gen_mask
                # net_back = mask_true[:, time_diff] * frames_back[:, t] + (1 - mask_true[:, time_diff]) * x_gen_back
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net_back = frames_back[:, (self.configs.input_length - 1)]
                # print("Itr: ", itr)
                # net_level_one = x_gen_level_one
                # net_level_two = x_gen_level_two
                # net_level_three = x_gen_level_three
                # net_level_four = x_gen_level_four
                net_level_one = mask_true_one[:, time_diff] * frames_level_one[:, t] + (1 - mask_true_one[:, time_diff]) * x_gen_level_one
                net_level_two = mask_true_two[:, time_diff] * frames_level_one[:, t] + (1 - mask_true_two[:, time_diff]) * x_gen_level_two
                net_level_three = mask_true_three[:, time_diff] * frames_level_one[:, t] + (1 - mask_true_three[:, time_diff]) * x_gen_level_three
                net_level_four = mask_true_four[:, time_diff] * frames_level_one[:, t] + (1 - mask_true_four[:, time_diff]) * x_gen_level_four
            # net_mask = frames_mask[:, t]
            # net_back = frames_back[:, t]
            frames_feature_level_one = net_level_one
            frames_feature_level_one_encoded = []
            frames_feature_level_two = net_level_two
            frames_feature_level_two_encoded = []
            frames_feature_level_three = net_level_three
            frames_feature_level_three_encoded = []
            frames_feature_level_four = net_level_four
            frames_feature_level_four_encoded = []
            for i in range(len(self.encoders_level_one)):
                frames_feature_level_one = self.encoders_level_one[i](frames_feature_level_one)
                frames_feature_level_one_encoded.append(frames_feature_level_one)
            for i in range(len(self.encoders_level_two)):
                frames_feature_level_two = self.encoders_level_two[i](frames_feature_level_two)
                frames_feature_level_two_encoded.append(frames_feature_level_two)
            for i in range(len(self.encoders_level_three)):
                frames_feature_level_three = self.encoders_level_three[i](frames_feature_level_three)
                frames_feature_level_three_encoded.append(frames_feature_level_three)
            for i in range(len(self.encoders_level_four)):
                frames_feature_level_four = self.encoders_level_three[i](frames_feature_level_four)
                frames_feature_level_four_encoded.append(frames_feature_level_four)
            if t == 0:
                for i in range(self.num_layers):
                    zeros_level_one = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_two = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_three = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_four = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t_level_one.append(zeros_level_one)
                    T_t_level_two.append(zeros_level_two)
                    T_t_level_three.append(zeros_level_three)
                    T_t_level_four.append(zeros_level_four)
            S_t_level_one = frames_feature_level_one
            # if t % 2 == 0:
            S_t_level_two = frames_feature_level_two
            # if t % 3 == 0:
            S_t_level_three = frames_feature_level_three

            S_t_level_four = frames_feature_level_four
            for i in range(self.num_layers):
                t_att_level_one = T_pre_level_one[i][-self.tau_one:]
                t_att_level_one = torch.stack(t_att_level_one, dim=0)
                s_att_level_one = S_pre_level_one[i][-self.tau_one:]
                s_att_level_one = torch.stack(s_att_level_one, dim=0)
                S_pre_level_one[i].append(S_t_level_one)

                t_att_level_two = T_pre_level_two[i][-self.tau_two:]
                t_att_level_two = torch.stack(t_att_level_two, dim=0)
                s_att_level_two = S_pre_level_two[i][-self.tau_two:]
                s_att_level_two = torch.stack(s_att_level_two, dim=0)
                S_pre_level_two[i].append(S_t_level_two)

                t_att_level_three = T_pre_level_three[i][-self.tau_three:]
                t_att_level_three = torch.stack(t_att_level_three, dim=0)
                s_att_level_three = S_pre_level_three[i][-self.tau_three:]
                s_att_level_three = torch.stack(s_att_level_three, dim=0)
                S_pre_level_three[i].append(S_t_level_three)

                t_att_level_four = T_pre_level_four[i][-self.tau_four:]
                t_att_level_four = torch.stack(t_att_level_four, dim=0)
                s_att_level_four = S_pre_level_four[i][-self.tau_four:]
                s_att_level_four = torch.stack(s_att_level_four, dim=0)
                S_pre_level_four[i].append(S_t_level_four)

                T_t_level_one[i], T_t_level_two[i], T_t_level_three[i],T_t_level_four[i], S_t_level_one, S_t_level_two, S_t_level_three, S_t_level_four = \
                    self.cell_list[i](T_t_level_one[i], T_t_level_two[i], T_t_level_three[i], T_t_level_four[i],
                                      S_t_level_one, S_t_level_two, S_t_level_three, S_t_level_four,
                                      t_att_level_one, s_att_level_one, t_att_level_two, s_att_level_two,
                                      t_att_level_three, s_att_level_three, t_att_level_four, s_att_level_four)
                T_pre_level_one[i].append(T_t_level_one[i])
                T_pre_level_two[i].append(T_t_level_two[i])
                T_pre_level_three[i].append(T_t_level_three[i])
                T_pre_level_four[i].append(T_t_level_four[i])
            out_level_one = S_t_level_one
            out_level_two = S_t_level_two
            out_level_three = S_t_level_three
            out_level_four = S_t_level_four
            # out = self.merge(torch.cat([T_t[-1], S_t], dim=1))
            frames_feature_decoded = []
            for i in range(len(self.decoders_level_one)):
                out_level_one = self.decoders_level_one[i](out_level_one)
                if self.configs.model_mode == 'recall':
                    out_level_one = out_level_one + frames_feature_level_one_encoded[-2 - i]
            for i in range(len(self.decoders_level_two)):
                out_level_two = self.decoders_level_two[i](out_level_two)
                if self.configs.model_mode == 'recall':
                    out_level_two = out_level_two + frames_feature_level_two_encoded[-2 - i]
            for i in range(len(self.decoders_level_three)):
                out_level_three = self.decoders_level_three[i](out_level_three)
                if self.configs.model_mode == 'recall':
                    out_level_three = out_level_three + frames_feature_level_three_encoded[-2 - i]
            for i in range(len(self.decoders_level_four)):
                out_level_four = self.decoders_level_four[i](out_level_four)
                if self.configs.model_mode == 'recall':
                    out_level_four = out_level_four + frames_feature_level_four_encoded[-2 - i]

            x_gen_level_one = self.srcnn_level_one(out_level_one)
            x_gen_level_two = self.srcnn_level_two(out_level_two)
            x_gen_level_three = self.srcnn_level_three(out_level_three)
            x_gen_level_four = self.srcnn_level_four(out_level_four)
            next_frames.append(x_gen_level_three)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
