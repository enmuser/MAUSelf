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
                        configs.filter_size_three, configs.stride, self.tau_one, self.tau_two, self.tau_three, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        encoders_mask = []
        encoder_mask = nn.Sequential()
        encoder_mask.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder_mask.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders_mask.append(encoder_mask)
        for i in range(n):
            encoder_mask = nn.Sequential()
            encoder_mask.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder_mask.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders_mask.append(encoder_mask)
        self.encoders_mask = nn.ModuleList(encoders_mask)


        encoders_back = []
        encoder_back = nn.Sequential()
        encoder_back.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder_back.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders_back.append(encoder_back)
        for i in range(n):
            encoder_back = nn.Sequential()
            encoder_back.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder_back.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders_back.append(encoder_back)
        self.encoders_back = nn.ModuleList(encoders_back)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        decoders_mask = []
        for i in range(n - 1):
            decoder_mask = nn.Sequential()
            decoder_mask.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder_mask.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders_mask.append(decoder_mask)

        if n > 0:
            decoder_mask = nn.Sequential()
            decoder_mask.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders_mask.append(decoder_mask)
        self.decoders_mask = nn.ModuleList(decoders_mask)

        decoders_back = []
        for i in range(n - 1):
            decoder_back = nn.Sequential()
            decoder_back.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder_back.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders_back.append(decoder_back)

        if n > 0:
            decoder_back = nn.Sequential()
            decoder_back.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders_back.append(decoder_back)
        self.decoders_back = nn.ModuleList(decoders_back)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.srcnn_mask = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.srcnn_back = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames_level_one, frames_level_two, frames_level_three, mask_true, itr):
        # print('ok')
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames_level_one.shape[0]
        height = frames_level_one.shape[3] // self.configs.sr_size
        width = frames_level_one.shape[4] // self.configs.sr_size
        frame_channels = frames_level_one.shape[2]
        next_frames = []
        T_t_level_one = []
        T_t_level_two = []
        T_t_level_three = []
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
                net_level_two = frames_level_two[:, t]
                net_level_three = frames_level_three[:, t]
            else:
                # time_diff = t - self.configs.input_length
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net_mask = mask_true[:, time_diff] * frames_mask[:, t] + (1 - mask_true[:, time_diff]) * x_gen_mask
                # net_back = mask_true[:, time_diff] * frames_back[:, t] + (1 - mask_true[:, time_diff]) * x_gen_back
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net_back = frames_back[:, (self.configs.input_length - 1)]
                # print("Itr: ", itr)
                net_level_one = x_gen_level_one
                net_level_two = x_gen_level_two
                net_level_three = x_gen_level_three
            # net_mask = frames_mask[:, t]
            # net_back = frames_back[:, t]
            frames_feature_level_one = net_level_one
            frames_feature_level_one_encoded = []
            frames_feature_level_two = net_level_two
            frames_feature_level_two_encoded = []
            frames_feature_level_three = net_level_three
            frames_feature_level_three_encoded = []
            for i in range(len(self.encoders)):
                frames_feature_level_one = self.encoders[i](frames_feature_level_one)
                frames_feature_level_one_encoded.append(frames_feature_level_one)
            for i in range(len(self.encoders_mask)):
                frames_feature_level_two = self.encoders_mask[i](frames_feature_level_two)
                frames_feature_level_two_encoded.append(frames_feature_level_two)
            for i in range(len(self.encoders_back)):
                frames_feature_level_three = self.encoders_back[i](frames_feature_level_three)
                frames_feature_level_three_encoded.append(frames_feature_level_three)
            if t == 0:
                for i in range(self.num_layers):
                    zeros_level_one = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_two = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_three = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t_level_one.append(zeros_level_one)
                    T_t_level_two.append(zeros_level_two)
                    T_t_level_three.append(zeros_level_three)
            S_t_level_one = frames_feature_level_one
            # if t % 2 == 0:
            S_t_level_two = frames_feature_level_two
            # if t % 3 == 0:
            S_t_level_three = frames_feature_level_three
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

                T_t_level_one[i], T_t_level_two[i], T_t_level_three[i], S_t_level_one, S_t_level_two, S_t_level_three = \
                    self.cell_list[i](T_t_level_one[i], T_t_level_two[i], T_t_level_three[i], S_t_level_one, S_t_level_two, S_t_level_three, t_att_level_one, s_att_level_one, t_att_level_two, s_att_level_two, t_att_level_three, s_att_level_three)
                T_pre_level_one[i].append(T_t_level_one[i])
                T_pre_level_two[i].append(T_t_level_two[i])
                T_pre_level_three[i].append(T_t_level_three[i])
            out_level_one = S_t_level_one
            out_level_two = S_t_level_two
            out_level_three = S_t_level_three
            # out = self.merge(torch.cat([T_t[-1], S_t], dim=1))
            frames_feature_decoded = []
            for i in range(len(self.decoders)):
                out_level_one = self.decoders[i](out_level_one)
                if self.configs.model_mode == 'recall':
                    out_level_one = out_level_one + frames_feature_level_one_encoded[-2 - i]
            for i in range(len(self.decoders_mask)):
                out_level_two = self.decoders_mask[i](out_level_two)
                if self.configs.model_mode == 'recall':
                    out_level_two = out_level_two + frames_feature_level_two_encoded[-2 - i]
            for i in range(len(self.decoders_back)):
                out_level_three = self.decoders_back[i](out_level_three)
                if self.configs.model_mode == 'recall':
                    out_level_three = out_level_three + frames_feature_level_three_encoded[-2 - i]

            x_gen_level_one = self.srcnn(out_level_one)
            x_gen_level_two = self.srcnn_mask(out_level_two)
            x_gen_level_three = self.srcnn_back(out_level_three)
            next_frames.append(x_gen_level_one)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
