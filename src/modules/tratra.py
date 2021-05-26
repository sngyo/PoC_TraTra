from logging import getLogger

import torch.nn as nn
from kornia import spatial_softargmax_2d

from .image_encoder import ImageEncoder
from .transformer import TF_Encoder

logger = getLogger(__name__)


class TraTra(nn.Module):
    def __init__(self, cfgs):
        super(TraTra, self).__init__()
        self.img_enc = ImageEncoder(
            cfgs.input_c, cfgs.img_enc_dim, cfgs.dropout, cfgs.img_enc_model
        )

        self.transformer = TF_Encoder(
            cfgs.img_enc_dim,
            cfgs.embed_dim,
            cfgs.nhead,
            cfgs.nhid,
            cfgs.nlayers,
            cfgs.device,
            cfgs.dropout,
        )

        self.img_h = cfgs.img_h
        self.img_w = cfgs.img_w

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        # self.device = device
        self.transformer.set_device(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor
            shape: (batch_size, 1, h, w)
        """
        # 1. image encoding
        x = self.img_enc(x)  # output: (batch_size, img_enc_dim, h, w)

        # 2. Generate heatmap by transformer encoder
        # output: (batch_size, em_dim, h', w'), (batch_size, em_dim)
        heatmaps, pt_state = self.transformer(x)
        _, _, heat_h, heat_w = heatmaps.shape

        # 3. Generate the coordinates  --> loss function
        # output: (batch_size, em_dim, x, y)
        coords = spatial_softargmax_2d(heatmaps, False)

        # 4. scale up for original image size
        coords[:, :, 0] = coords[:, :, 0] * (self.img_w / heat_w)
        coords[:, :, 1] = coords[:, :, 1] * (self.img_h / heat_h)
        return heatmaps, coords, pt_state


if __name__ == "__main__":
    # NOTE: this is only for initial debug purpose
    #       remove dot "." from `from .XXX import yyy`
    # import torch
    from easydict import EasyDict
    from torchinfo import summary

    cfgs = EasyDict()
    cfgs.input_c = 1  # grayscale
    cfgs.img_enc_dim = 256
    cfgs.embed_dim = 256
    cfgs.nhead = 8
    cfgs.nhid = 256
    cfgs.nlayers = 3
    cfgs.dropout = 0.3
    cfgs.img_enc_model = "UNet"
    cfgs.img_h = 32
    cfgs.img_w = 32

    model = TraTra(cfgs).to("cuda:0")
    # with torch.no_grad():
    #     x = torch.zeros(4, 1, 32, 32).to("cuda:0")
    #     h, c, p = model(x)
    #     print(h.shape)
    #     print(c.shape)
    #     print(p.shape)
    print(summary(model, input_size=(4, 1, 32, 32)))
