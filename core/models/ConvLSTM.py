import torch
from torch import nn
from torch.autograd import Variable

from core.models.ConvLSTMCell import ConvLSTMCell


class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, device, kernel_size=(3, 3),**kwargs):
        super().__init__()
        cell_list = []
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = True
        self.device = device
        self.hidden = None
        for i in range(0, self.num_layers): # num_layers = 4 ,[0,1,2,3]
            cur_input_size = self.hidden_size[0] if i == 0 else self.hidden_size[i - 1]# predictor:  hidden_size =[128, 128, 128, 128] => cur_input_size = 128
            cell_list.append(ConvLSTMCell(
                    input_size=cur_input_size, # cur_input_size = 128
                    hidden_size=self.hidden_size[i], # hidden_size = [128,128,128,128]
                    kernel_size=self.kernel_size[i], # kernel_size = [(3,3),(3,3),(3,3),(3,3)]
                    bias=self.bias # True
                )
            ) # input_size = 138, hidden_size = 128, kernel_size = (3,3)//input_size = 266, hidden_size = 128, kernel_size = (3,3)//input_size=522, hidden_size = 128, kernel_size = (3,3)
        self.input_conv = nn.Conv2d(self.input_size, self.hidden_size[0], kernel_size=kernel_size[0], padding=1)
        self.cell_list = nn.ModuleList(cell_list)
        self.out_proj = nn.Sequential( # hidden_size = 128, output_size = [128,256,512]
                nn.Conv2d(in_channels=hidden_size[0], out_channels=output_size, kernel_size=3, padding=1),
                nn.Tanh()
            )
        return

    def forward(self, x, hidden_state=None):
        B, C, H, W = x.shape # 16 * 138 * 16 * 16 //// 16 * 266 * 8 * 8
        if self.hidden is None:
            hidden_list = []
            for i in range(self.num_layers):
                hidden_list.append(self.init_hidden(batch_size=B, input_size=(H, W),device=self.device))
            self.hidden = hidden_list

        cur_input = self.input_conv(x) # 16 * 138 * 16 * 16 => 16 * 128 * 16 * 16
        # iterating through layers # 16 * 266 * 8 * 8 => 16 * 128 * 8 * 8
        for i in range(self.num_layers):
            self.hidden[i] = self.cell_list[i](x=cur_input, state=self.hidden[i])
            cur_input = self.hidden[i][0]  # cur layer output is next layer input

        output = self.out_proj(cur_input) # 16 * 128 * 16 * 16
        self.last_output = output # 16 * 128 * 16 * 16
        return output

    def init_hidden(self, batch_size, input_size, device):
        """ Initializing the hidden state of the cell """
        height, width = input_size # input_size = (16,16), batch_size = 16, hidden_size = 128
        state = (Variable(torch.zeros(batch_size, self.hidden_size[0], height, width, device=device)),
                 Variable(torch.zeros(batch_size, self.hidden_size[0], height, width, device=device)))
        return state # ((16,128,16,16),(16,128,16,16)) // ((16,128,8,8),(16,128,8,8)) //  ((16,128,4,4),(16,128,4,4)) //// prior/post: ((16,64,16,16),(16,64,16,16)) // ((16,64,8,8),(16,64,8,8)) //  ((16,64,4,4),(16,64,4,4))
