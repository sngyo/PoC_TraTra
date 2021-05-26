import math
from logging import getLogger

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logger = getLogger(__name__)


# ref: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionalEncoding2d(nn.Module):
    def __init__(
        self, d_model, device, dropout=0.1, max_height=32, max_width=32
    ):
        """
        Parameters
        ----------
        d_model: the dimension of the model
        device: str
        dropout:
        max_height:
        max_width
        """
        super(PositionalEncoding2d, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 2d version
        # pe.shape --> (d_model, max_height, max_width)
        pe = torch.zeros(d_model, max_height, max_width).to(device)
        d_model = d_model // 2  # for h, w dividing into 2
        pos_h = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        pos_w = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )

        pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, max_height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, max_height, 1)
        )

        pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term)
            .transpose(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, max_width)
        )
        pe[d_model + 1::2, :, :] = (
            torch.cos(pos_h * div_term)
            .transpose(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, max_width)
        )

        self.register_buffer("pe", pe)

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        self.pe.to(device)
        logger.info(f"PosEncode2D device updated to {self.device}")

    def forward(self, x):
        x = x + self.pe[:, : x.size(2), : x.size(3)]
        x = self.dropout(x)
        return x


class TF_Encoder(nn.Module):
    def __init__(
        self, nchannel, ninp, nhead, nhid, nlayers, device, dropout=0.2
    ):
        """
        Parameters
        ----------
        nchannel: the number of channles from CNN
        ninp: embedding dimension
        nhid: the dimension of the feedforward network model
        nlayers: the number of nn.TransoformerEncoderLayer
        nhead: the number of heads in the multiheadattention models
        device: str
        dropout: dropout value
        """
        super(TF_Encoder, self).__init__()
        self.device = device
        self.ninp = ninp

        # 2d position encoding
        self.pos_encoder = PositionalEncoding2d(ninp, device, dropout)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.tf_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(nchannel, ninp)

        # decoder output dim should equal to the lenght of time-step
        self.decoder = nn.Linear(ninp, ninp)
        self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        ini_range = 0.1
        self.encoder.weight.data.uniform_(-ini_range, ini_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-ini_range, ini_range)

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        self.device = device
        self.pos_encoder.set_device(device)
        logger.info(f"TF_Encoder device updated to {self.device}")

    def forward(self, x):
        """
        Parameters
        ----------
        x: Shape: (Batch, Channel, H, W)

        Return
        ------
        x : torch.tensor
            (Batch, n_inp, H, W)
        point_state : torch.tensor
            (Batch, n_inp)
        """
        (batch_size, in_c, h, w) = x.shape
        x = self.encoder(x.permute(0, 2, 3, 1))  # --> (Batch, H, W, n_inp)
        x = self.pos_encoder(x.permute(0, 3, 1, 2))  # --> (Batch, n_inp, H, W)
        x = x.reshape(batch_size, in_c, -1)  # --> (Batch, n_inp, H*W)

        # ideal output will contain stroke_id such as
        # [1, 1, 1, ..., 2, 2, 2, ..., 3, 3, 3, ... 0, 0, 0]
        tmp = torch.zeros(batch_size, in_c, x.shape[2] + 1).to(self.device)
        tmp[:, :, :-1] = x
        x = tmp  # --> (Batch, n_inp, H*W+1)

        x = self.tf_encoder(x.permute(2, 0, 1))  # --> (H*W+1, Batch, n_inp)
        x = self.decoder(x)  # --> (H*W+1, Batch, n_inp)

        # split x and point_state [CLS]
        point_state = self.sigmoid(x[-1])  # --> (Batch, n_inp)

        x = x[:-1]  # --> (H*W, Batch, n_inp)
        x = x.permute(1, 2, 0).reshape(batch_size, -1, h, w)
        x = self.softmax(x.view(*x.size()[:2], -1)).view_as(x)
        return x, point_state


if __name__ == "__main__":
    # Debug purpose only
    # from torchinfo import summary
    n_channels = 256
    em_dim = 256
    nhid = 256
    nlayers = 3
    nhead = 4
    dropout = 0.2

    # model = PositionalEncoding2d(256)
    model = TF_Encoder(n_channels, em_dim, nhead, nhid, nlayers, dropout)
    dummy = torch.autograd.Variable(torch.randn(3, 256, 32, 32))
    dummy_out, dummy_pt_state = model(dummy)

    print(dummy_out.shape)
    print(dummy_pt_state.shape)
    # model(torch.randn(3, 256, 32, 32))
    # print(summary(model, (256, 32, 32)))
