from copy import deepcopy

import torch
import torch.nn as nn
from core.layers.MAUCell import MAUCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
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

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

        n = 4
        diff_encoders = []
        diff_encoder = nn.Sequential()
        diff_encoder.add_module(name='diff_encoder_t_conv{0}'.format(-1),
                                # frame_channel = 1, num_hidden = 64
                                module=nn.Conv2d(in_channels=self.frame_channel,
                                                 out_channels=self.num_hidden[0],
                                                 stride=1,
                                                 padding=0,
                                                 kernel_size=1))
        diff_encoder.add_module(name='diff_encoder_relu_t_{0}'.format(-1),
                                module=nn.LeakyReLU(0.2))
        diff_encoders.append(diff_encoder)
        for i in range(n):
            # 每次图片大小减掉一倍
            diff_encoder = nn.Sequential()
            diff_encoder.add_module(name='diff_encoder_t{0}'.format(i),
                                    # in_channels = 64, out_channels = 64, stride = (2,2),padding = (1,1), kernel_size= (3,3)
                                    # outshape = 每次图片大小减掉一半
                                    module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                     out_channels=self.num_hidden[0],
                                                     stride=(2, 2),
                                                     padding=(1, 1),
                                                     kernel_size=(3, 3)
                                                     ))
            diff_encoder.add_module(name='diff_encoder_t_relu{0}'.format(i),
                                    module=nn.LeakyReLU(0.2))
            diff_encoders.append(diff_encoder)
        diff_encoder = nn.Sequential()
        diff_encoder.add_module(name='diff_encoder_t_conv{0}'.format(-1),
                                # frame_channel = 1, num_hidden = 64
                                module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                 out_channels=20,
                                                 stride=1,
                                                 padding=0,
                                                 kernel_size=1))
        diff_encoder.add_module(name='diff_encoder_relu_t_{0}'.format(-1),
                                module=nn.LeakyReLU(0.2))
        diff_encoders.append(diff_encoder)
        self.diff_encoders = nn.ModuleList(diff_encoders)

        n = 4
        pred_diff_encoders = []
        pred_diff_encoder = nn.Sequential()
        pred_diff_encoder.add_module(name='pred_diff_encoder_t_conv{0}'.format(-1),
                                     # frame_channel = 1, num_hidden = 64
                                     module=nn.Conv2d(in_channels=self.frame_channel,
                                                      out_channels=self.num_hidden[0],
                                                      stride=1,
                                                      padding=0,
                                                      kernel_size=1))
        pred_diff_encoder.add_module(name='pred_diff_encoder_relu_t_{0}'.format(-1),
                                     module=nn.LeakyReLU(0.2))
        pred_diff_encoders.append(pred_diff_encoder)
        for i in range(n):
            # 每次图片大小减掉一倍
            pred_diff_encoder = nn.Sequential()
            pred_diff_encoder.add_module(name='pred_diff_encoder_t{0}'.format(i),
                                         # in_channels = 64, out_channels = 64, stride = (2,2),padding = (1,1), kernel_size= (3,3)
                                         # outshape = 每次图片大小减掉一半
                                         module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                          out_channels=self.num_hidden[0],
                                                          stride=(2, 2),
                                                          padding=(1, 1),
                                                          kernel_size=(3, 3)
                                                          ))
            pred_diff_encoder.add_module(name='pred_diff_encoder_t_relu{0}'.format(i),
                                         module=nn.LeakyReLU(0.2))
            pred_diff_encoders.append(pred_diff_encoder)
        pred_diff_encoder = nn.Sequential()
        pred_diff_encoder.add_module(name='pred_diff_encoder_t_conv{0}'.format(-1),
                                     # frame_channel = 1, num_hidden = 64
                                     module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                      out_channels=20,
                                                      stride=1,
                                                      padding=0,
                                                      kernel_size=1))
        pred_diff_encoder.add_module(name='pred_diff_encoder_relu_t_{0}'.format(-1),
                                     module=nn.LeakyReLU(0.2))
        pred_diff_encoders.append(pred_diff_encoder)
        self.pred_diff_encoders = nn.ModuleList(pred_diff_encoders)

    def forward(self, frames, mask_true):
        # print('ok')
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)
        empty_lists = []  # num_hierarch = 3
        out_dict = {"mu_post": deepcopy(empty_lists), "logvar_post": deepcopy(empty_lists),
                    "mu_prior": deepcopy(empty_lists), "logvar_prior": deepcopy(empty_lists)}
        for t in range(self.configs.total_length - 1):
            if t >= self.configs.input_length:
                diff_frame = frames[:, t+1] - frames[:, t-2]
                # for i in range(len(self.diff_encoders)):
                #     diff_frame = self.diff_encoders[i](diff_frame)
                # diff_mu_post, diff_logvar_post = torch.chunk(diff_frame, chunks=2, dim=1)
                out_dict["mu_post"].append(diff_frame)
                out_dict["logvar_post"].append(diff_frame)
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                # time_diff = t - self.configs.input_length
                # net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                net = x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t
            # out = self.merge(torch.cat([T_t[-1], S_t], dim=1))
            frames_feature_decoded = []
            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            if t >= self.configs.input_length:
                pred_diff_frame = x_gen - next_frames[-3]
                # for i in range(len(self.pred_diff_encoders)):
                #     pred_diff_frame = self.pred_diff_encoders[i](pred_diff_frame)
                # diff_mu_prior, logvar_prior = torch.chunk(pred_diff_frame, chunks=2, dim=1)
                out_dict["mu_prior"].append(pred_diff_frame)
                out_dict["logvar_prior"].append(pred_diff_frame)
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames, out_dict
