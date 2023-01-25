import torch
from torch import nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Singel Convolutional LSTM cell. Implements the basic LSTM gates,
    but using Convolutional layers, instead of Fully Connected
    Adapted from: https://github.com/ndrplz/ConvLSTM_pytorch

    Args:
    -----
    input_size: int
        Number of channels of the input
    hidden_size: int
        Number of channels of hidden state.
    kernel_size: int or tuple
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """

    def __init__(self, input_size, hidden_size, kernel_size=(3, 3), bias=True):
        """ Module initializer """ # 1.input_size = 64, hidden_size = 64, kernel_size=(3, 3) //// predictor: 1.input_size = 128, hidden_size = 128, kernel_size=(3, 3)
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size # kernel_size=(3, 3)
        assert len(kernel_size) == 2, f"Kernel size {kernel_size} has wrong shape"
        super().__init__()
        self.input_size = input_size # input_size = [64,64,64] //// predictor: input_size = [128,128,128,128]
        self.hidden_size = hidden_size # hidden_size = [64,64,64] ////  predictor: hidden_size = [128,128,128,128]

        self.kernel_size = kernel_size # kernel_size=(3, 3)
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # padding = (1,1)
        self.bias = bias # True

        self.conv = nn.Conv2d( #
                in_channels=self.input_size + self.hidden_size, # in_channels = [64,64,64] + [64,64,64] = [128,128,128] //// predictor: in_channels = [128,128,128,128] + [128,128,128,128] = [256,256,256,256]
                out_channels=4 * self.hidden_size,# out_channels = 256 //// predictor: out_channels = 4 * [128,128,128,128] = [512,512,512,512]
                kernel_size=self.kernel_size,# kernel_size=(3, 3)
                padding=self.padding,# padding = (1,1)
                bias=self.bias# True
            )

        self.hidden = None
        return

    def forward(self, x, state=None):
        """ x: 16 * 64 * 4 * 4 //// x = 16 * 128 * 16 * 16
        Forward pass of an input through the ConvLSTM cell

        Args:
        -----
        x: torch Tensor
            Feature maps to forward through ConvLSTM Cell. Shape is (B, input_size, H, W)
        state: tuple
            tuple containing the hidden and cell state. Both have shape (B, hidden_size, H, W)
        """
        if state is None:
            state = self.init_hidden(batch_size=x.shape[0], input_size=x.shape[-2:])
        hidden_state, cell_state = state
        # hidden_state = 16 * 64 * 4 * 4 / cell_state = 16 * 64 * 4 * 4 //// hidden_state = 16 * 128 * 16 * 16 / cell_state = 16 * 128 * 16 * 16
        # joinly computing all convs by stacking and spliting across channel dim.
        input = torch.cat([x, hidden_state], dim=1) # => 16 * 128 * 4 * 4 //// =>  16 * 256 * 16 * 16
        out_conv = self.conv(input) # out_conv = 16 * 256 * 4 * 4 <= 16 * 128 * 4 * 4 ////  out_conv = 16 * 512 * 16 * 16 <= 16 * 256 * 16 * 16
        cc_i, cc_f, cc_o, cc_g = torch.split(out_conv, self.hidden_size, dim=1) # 一分为四 #
        # cc_i, cc_f, cc_o, cc_g = 16 * 64 * 4 * 4，16 * 64 * 4 * 4，16 * 64 * 4 * 4，16 * 64 * 4 * 4
        # computing input, forget, update and output gates  # cc_i, cc_f, cc_o, cc_g = 16 * 128 * 16 * 16，16 * 128 * 16 * 16，16 * 128 * 16 * 16，16 * 128 * 16 * 16
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # updating hidden and cell state
        updated_cell_state = f * cell_state + i * g
        updated_hidden_state = o * torch.tanh(updated_cell_state)
        return updated_hidden_state, updated_cell_state # [16 * 64 * 4 * 4,16 * 64 * 4 * 4] //// [16 * 128 * 16 * 16,16 * 128 * 16 * 16]

    def init_hidden(self, batch_size, input_size, device):
        """ Initializing the hidden state of the cell """
        height, width = input_size # input_size = (16,16), batch_size = 16, hidden_size = 128
        state = (Variable(torch.zeros(batch_size, self.hidden_size, height, width, device=device)),
                 Variable(torch.zeros(batch_size, self.hidden_size, height, width, device=device)))
        return state # ((16,128,16,16),(16,128,16,16)) // ((16,128,8,8),(16,128,8,8)) //  ((16,128,4,4),(16,128,4,4)) //// prior/post: ((16,64,16,16),(16,64,16,16)) // ((16,64,8,8),(16,64,8,8)) //  ((16,64,4,4),(16,64,4,4))
