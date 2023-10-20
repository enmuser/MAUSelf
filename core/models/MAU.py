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
        self.tau = configs.tau
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
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride, self.tau, self.cell_mode)
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

    def forward(self, frames, frames_mask, frames_back, img_gen_f, img_gen_b, mask_true,itr):
        # print('ok')
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_t_level_one = []
        T_t_level_two = []
        T_pre = []
        S_pre = []
        T_pre_level_one = []
        S_pre_level_one = []
        T_pre_level_two = []
        S_pre_level_two = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            tmp_t_level_one = []
            tmp_s_level_one = []
            tmp_t_level_two = []
            tmp_s_level_two = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_t_level_one.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s_level_one.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_t_level_two.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s_level_two.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)
            T_pre_level_one.append(tmp_t_level_one)
            S_pre_level_one.append(tmp_s_level_one)
            T_pre_level_two.append(tmp_t_level_two)
            S_pre_level_two.append(tmp_s_level_two)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
                net_mask = frames_mask[:, t]
                net_back = frames_back[:, t]
            else:
                # time_diff = t - self.configs.input_length
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net_mask = mask_true[:, time_diff] * frames_mask[:, t] + (1 - mask_true[:, time_diff]) * x_gen_mask
                # net_back = mask_true[:, time_diff] * frames_back[:, t] + (1 - mask_true[:, time_diff]) * x_gen_back
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # net_back = frames_back[:, (self.configs.input_length - 1)]
                # print("Itr: ", itr)
                if itr <= self.train_level_base_line:
                    net_mask = frames_mask[:, t]
                    net_back = frames_back[:, t]
                elif itr <= (self.train_level_base_line + 35000):
                    if t <= 17:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = x_gen_mask
                        net_back = x_gen_back
                elif itr <= (self.train_level_base_line + 85000):
                    if t <= 16:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = x_gen_mask
                        net_back = x_gen_back
                elif itr <= (self.train_level_base_line + 120000):
                    if t <= 15:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = x_gen_mask
                        net_back = x_gen_back
                elif itr <= (self.train_level_base_line + 155000):
                    if t <= 14:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = img_gen_f[:, t]
                        net_back = img_gen_b[:, t]
                elif itr <= (self.train_level_base_line + 600000):
                    if t <= 13:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = img_gen_f[:, t]
                        net_back = img_gen_b[:, t]
                elif itr <= (self.train_level_base_line + 750000):
                    if t <= 12:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = img_gen_f[:, t]
                        net_back = img_gen_b[:, t]
                elif itr <= (self.train_level_base_line + 900000):
                    if t <= 11:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = img_gen_f[:, t]
                        net_back = img_gen_b[:, t]
                elif itr <= (self.train_level_base_line + 1050000):
                    if t <= 10:
                        net_mask = frames_mask[:, t]
                        net_back = frames_back[:, t]
                    else:
                        net_mask = img_gen_f[:, t]
                        net_back = img_gen_b[:, t]
                else:
                    net_mask = img_gen_f[:, t]
                    net_back = img_gen_b[:, t]
                net = x_gen
            # net_mask = frames_mask[:, t]
            # net_back = frames_back[:, t]
            frames_feature = net
            frames_feature_encoded = []
            frames_feature_mask = net_mask
            frames_feature_mask_encoded = []
            frames_feature_back = net_back
            frames_feature_back_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            for i in range(len(self.encoders_mask)):
                frames_feature_mask = self.encoders_mask[i](frames_feature_mask)
                frames_feature_mask_encoded.append(frames_feature_mask)
            for i in range(len(self.encoders_back)):
                frames_feature_back = self.encoders_back[i](frames_feature_back)
                frames_feature_back_encoded.append(frames_feature_back)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_one = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    zeros_level_two = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t.append(zeros)
                    T_t_level_one.append(zeros_level_one)
                    T_t_level_two.append(zeros_level_two)
            S_t = frames_feature
            # if t % 2 == 0:
            S_t_level_one = frames_feature_mask
            # if t % 3 == 0:
            S_t_level_two = frames_feature_back
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)

                t_att_level_one = T_pre_level_one[i][-self.tau:]
                t_att_level_one = torch.stack(t_att_level_one, dim=0)
                s_att_level_one = S_pre_level_one[i][-self.tau:]
                s_att_level_one = torch.stack(s_att_level_one, dim=0)
                S_pre_level_one[i].append(S_t_level_one)

                t_att_level_two = T_pre_level_two[i][-self.tau:]
                t_att_level_two = torch.stack(t_att_level_two, dim=0)
                s_att_level_two = S_pre_level_two[i][-self.tau:]
                s_att_level_two = torch.stack(s_att_level_two, dim=0)
                S_pre_level_two[i].append(S_t_level_two)

                T_t[i], T_t_level_one[i], T_t_level_two[i], S_t, S_t_level_one, S_t_level_two = \
                    self.cell_list[i](T_t[i], T_t_level_one[i], T_t_level_two[i], S_t, S_t_level_one, S_t_level_two, t_att, s_att, t_att_level_one, s_att_level_one, t_att_level_two, s_att_level_two)
                T_pre[i].append(T_t[i])
                T_pre_level_one[i].append(T_t_level_one[i])
                T_pre_level_two[i].append(T_t_level_two[i])
            out = S_t
            out_mask = S_t_level_one
            out_back = S_t_level_two
            # out = self.merge(torch.cat([T_t[-1], S_t], dim=1))
            frames_feature_decoded = []
            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]
            for i in range(len(self.decoders_mask)):
                out_mask = self.decoders_mask[i](out_mask)
                if self.configs.model_mode == 'recall':
                    out_mask = out_mask + frames_feature_mask_encoded[-2 - i]
            for i in range(len(self.decoders_back)):
                out_back = self.decoders_back[i](out_back)
                if self.configs.model_mode == 'recall':
                    out_back = out_back + frames_feature_back_encoded[-2 - i]

            x_gen = self.srcnn(out)
            x_gen_mask = self.srcnn_mask(out_mask)
            x_gen_back = self.srcnn_back(out_back)
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
