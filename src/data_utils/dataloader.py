import logging
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils import get_specified_ext_fnames

logger = logging.getLogger(__name__)


class TraTraDataset(Dataset):
    def __init__(
            self,
            unp_name_list,
            unp_dir_name,
            max_len,
            resized_shape,
            transform=None
    ):
        """
        * TODO HDF5 version for efficient memory usage!

        Parameters
        ----------
        unp_name_list: str
            unp file paths
        unp_dir_name : str
        max_len : int
            max length of prediction / target (fixed by transformer dim)
        resized_shape : np.array
            np.array([h, w])
        transform : torch.transforms, default is None
        """
        super().__init__()
        self.unp_name_list = unp_name_list
        self.unp_dir_name = unp_dir_name
        self.max_len = max_len
        self.resized_shape = resized_shape
        self.transform = transform

        # load once on memory
        self.unp_dict = {}
        self.img_dict = {}
        self.scl_dict = {}

    def __len__(self):
        return len(self.unp_name_list)

    def __load_img(self, uname):
        # Need to save images in 'fig/' in the same parent directory of
        # self.unp_dir_name
        fname = uname.replace(self.unp_dir_name, "fig").replace(".unp", ".png")

        try:
            image = self.img_dict[fname]
            scale = self.scl_dict[fname]

        except KeyError:
            if not os.path.isfile(fname):
                logger.error(f"Cannot find {fname} while data loading")
                sys.exit()

            image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            scale = self.resized_shape / image.shape
            self.scl_dict[fname] = scale

            if self.transform is not None:
                # convert to PIL image
                image = Image.fromarray(image.astype(np.uint8))
                image = self.transform(image)
            self.img_dict[fname] = image
        return image, scale

    def __load_unp(self, fname, scale):
        # original unp does not support for now
        """
        Parameters
        ----------
        scale: np.array([height_scale, width_scale]) == (y, x)

        Return
        ------
        data: [[x0, y0, 0], [x1, y2, 0],,, [xn, yn, 1], [-1, -1, -1],,,]
            data consists of point that is represented by x_pos, y_pos and
            stroke_id
        """
        try:
            pad_data = self.unp_dict[fname]

        except KeyError:
            with open(fname, "r", encoding="latin-1") as f:
                lines = f.readlines()

                if lines[0] != "X Y\n":
                    logger.error(f"{fname} is not readable")
                    sys.exit()

                s_id = 0  # stroke_id
                data = []
                for line in lines[1:]:
                    if "=====" in line:
                        data[-1][-1] = 1  # pen_up
                        s_id += 1
                    else:
                        coord = line.split()
                        data.append([float(coord[0]), float(coord[1]), s_id])

            data = torch.Tensor(data)
            data[:, 0] = data[:, 0] * scale[1]  # resized coords
            data[:, 1] = data[:, 1] * scale[0]  # resized coords
            # padding with [-1, -1, -1]
            if len(data) > self.max_len:
                logger.error(f"{fname} has more than {self.max_len} points")
                sys.exit()
            pad_data = torch.ones(self.max_len, 3) * -1
            pad_data[:len(data)] = data
            self.unp_dict[fname] = pad_data
        return pad_data

    def __getitem__(self, idx):
        fname = self.unp_name_list[idx]
        image, scale = self.__load_img(fname)
        target = self.__load_unp(fname, scale)
        return image, target, fname


class InfTraTraDataset(Dataset):
    """
    Inference mode usage only
    """

    def __init__(self, path, transform=None):
        """
        Parameters
        ----------
        path : str
            directory path or specific image path
        transform : torch.transform, default is None
        """
        super().__init__()
        self.image_path_list = self.__get_img_path(path)
        self.transform = transform
        logger.info(f"{self.__len__()} image files found!")

    def __get_img_path(self, path):
        # 1. image path
        if os.path.isfile(path):
            # file type check is entrusted cv2.imread()
            image_path_list = [path]
        elif os.path.isdir(path):
            logger.info("Currently, TraTra loads only `.png` files.")
            image_path_list = get_specified_ext_fnames(path, ".png")
        else:
            logger.error(f"configs.test_data_path [{path}] is incorrect")
            sys.exit()
        return image_path_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        iname = self.image_path_list[idx]
        image = cv2.imread(iname, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return image, iname


def get_dataloader(configs):
    all_unp_fnames = get_specified_ext_fnames(
        configs.unp_dir_path, ".unp", exclude=configs.exclude_files
    )

    # TODO: dataset should be splitted at first and save filenames?
    y_train, y_test = train_test_split(
        all_unp_fnames, test_size=configs.test_size, random_state=configs.seed
    )
    y_train, y_valid = train_test_split(
        y_train, test_size=configs.valid_size, random_state=configs.seed
    )

    logger.info(f"train dataset has {len(y_train)} files")
    logger.info(f"valid dataset has {len(y_valid)} files")
    logger.info(f"test dataset has {len(y_test)} files")

    # transform
    train_transforms = transforms.Compose([
        transforms.Resize([configs.img_h, configs.img_w]),
        transforms.ToTensor(),
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize([configs.img_h, configs.img_w]),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize([configs.img_h, configs.img_w]),
        transforms.ToTensor(),
    ])

    # dataset, dataloader
    unp_dir = os.path.basename(os.path.dirname(configs.unp_dir_path))
    train_dataset = TraTraDataset(
        y_train,
        unp_dir,
        configs.max_output_len,
        np.array([configs.img_h, configs.img_w]),
        transform=train_transforms,
    )
    valid_dataset = TraTraDataset(
        y_valid,
        unp_dir,
        configs.max_output_len,
        np.array([configs.img_h, configs.img_w]),
        transform=valid_transforms,
    )
    test_dataset = TraTraDataset(
        y_test,
        unp_dir,
        configs.max_output_len,
        np.array([configs.img_h, configs.img_w]),
        transform=test_transforms,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=configs.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=configs.batch_size, shuffle=False
    )
    return train_loader, valid_loader, test_loader


def get_inf_dataloader(configs):
    """
    TODO: create config.inf_batch_size
    """
    inf_transforms = transforms.Compose([
        transforms.Resize([configs.img_h, configs.img_w]),
        transforms.ToTensor(),
    ])

    inf_dataset = InfTraTraDataset(configs.test_data_path, inf_transforms)
    inf_dataloader = DataLoader(
        inf_dataset, batch_size=configs.batch_size, shuffle=False
    )
    return inf_dataloader
