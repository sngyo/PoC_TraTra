from logging import getLogger

import torch.nn as nn

from .unet_parts import DoubleConv, Down, OutConv, Up

logger = getLogger(__name__)


# ref: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class UNet(nn.Module):
    def __init__(self, in_c, out_c, bilinear=True):
        super(UNet, self).__init__()
        factor = 2 if bilinear else 1

        # Down Part
        self.inconv = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)

        # Up Part
        # self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outconv = OutConv(64, out_c)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x


class REDNet(nn.Module):
    def __init__(self):
        super(REDNet, self).__init__()
        logger.error("REDNet is not implemented")
        pass

    def forward(self):
        pass


class CNN(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.3):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # 1st block (1, 128, 128)  --> (32, 128, 128)
            nn.Conv2d(in_c, out_c // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_c // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 2nd block (32, 128, 128) --> (128, 64, 64)
            nn.Conv2d(out_c // 8, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_c // 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 3rd block (128, 64, 64) --> (256, 32, 32)
            nn.Conv2d(out_c // 2, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # maybe linear from 64 to 256
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.3, img_enc_model="CNN"):
        """
        Parameters
        ----------
        in_c :
        out_c :
        dropout : float, default is 0.3
        img_enc_model : str
            ["CNN", "UNet"]
        """
        super(ImageEncoder, self).__init__()
        if img_enc_model == "CNN":
            self.model = CNN(in_c, out_c, dropout)
        elif img_enc_model == "UNet":
            self.model = UNet(in_c, out_c, dropout)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Debug purpose only
    from torchsummary import summary

    model = ImageEncoder(1, 64, img_enc_model="UNet")
    # print(model)

    image_size = 32
    print(summary(model, (1, image_size, image_size)))
