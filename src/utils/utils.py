import collections
import os
import random
import sys
from logging import getLogger

import cv2
import numpy as np
import torch

logger = getLogger(__name__)


def get_specified_ext_fnames(root_dir, ext, exclude=None):
    """
    Parameters
    ----------
    root_dir:
    ext:
    exclude: str (default: None)
        File path that contain unp file names to be excluded.
    """
    fnames = []
    if ext[0] != ".":
        ext = "." + ext

    exclude_files = []
    if (exclude is not None) and check_file_exist(exclude):
        with open(exclude, "r") as f:
            lines = f.readlines()

            for line in lines:
                # skip comment line and blank line
                if line[0] == "#" or line == "\n":
                    continue
                else:
                    exclude_files.append(line.replace("\n", ""))

    for curdir, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1] == ext:
                fname = os.path.join(curdir, file)
                if fname not in exclude_files:
                    fnames.append(fname)
    return sorted(fnames)


def seed_everything(seed=76):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_file_exist(fname, exit=True):
    """
    Check file existence

    Parameters
    ----------
    fname: string
        file path
    exit: bool (default: True)
        if `fname` is not found and `exit` is True, script exits by sys.exit()
        if `exit` is False, just return False instead of quitting script
    """
    if os.path.isfile(fname):
        return True
    elif exit:
        logger.error(f"{fname} does not exist")
        sys.exit()
    else:
        return False


def count_parameters(model):
    """
    Counting nn.Module trainable parameters

    ref: https://discuss.pytorch.org/t/
                how-do-i-check-the-number-of-parameters-of-a-model/4325/8

    Parameters
    ----------
    model : nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model, optimizer, model_path, device, is_dict=False):
    """
    Load PyTorch model

    TODO: DEBUG: multi-gpu mode loading data
        >>> sample code
            trained_weights = 'PATH_TO_TRAINED_WEIGHT.pth'
            state_dict = torch.load(
                trained_weights, map_location=lambda storage, loc: storage
            )
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            self.load_state_dict(new_state_dict)

    Parameters
    ----------
    model : torch.nn.Module
    optimizer : torch.optim or None
    model_path : str
    device : str
    is_dict : bool
        True when the save model is collections.OrderedDict

    Returns
    -------
    model : torch.nn.Module
    optimizer : torch.optim or None
    """
    model.to("cpu")

    if is_dict:
        # if we saved model like following (officially recommended way)
        # torch.save(model.state_dict(), savepath)
        if isinstance(torch.load(model_path), collections.OrderedDict):
            model.load_state_dict(torch.load(model_path))
            logger.info("optimizer info was not saved!")
        else:
            logger.error(
                f"is_dict was True, but {model_path} is not torch.state_dict"
            )
    else:
        # torch.save(model, savepath)
        model = torch.load(model_path, map_location="cpu")
        if optimizer is not None:
            optimizer.load_state_dict(model.info_dict["optimizer"])

    model = model.to(device)
    logger.info(f"Loading {model_path}")

    n_params = count_parameters(model)
    logger.info(f"Trainable Params: {n_params}")

    try:
        # If model has `device` attribute, update device_id
        model.set_device(device)
    except AttributeError:
        logger.warning(f"set_device() function cannot found ({model_path})")
        pass
    return model, optimizer


def plot_output(
        image,  # base image (input image)
        coords_data,  # rescaled and cut off the uncesecssary part
        savepath,
        plot_type="line",  # 'line', 'point', 'poly'
        color=[255 / 255, 140 / 255, 0],  # RGB color
        thickness=1,
        grad=False,  # color gradation
):
    """
    Plot the output function
    TODO: how to treat multi-stroke?
    TODO: docstring
    TODO: FIXME: when grad is True, you should reinit color every time
    TODO: scaling save
    FIXME: B006 Do not mutable data structure for argument defaults
    """
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if plot_type == "line":
        prev_p = coords_data[0, 0], coords_data[0, 1]
        for p in coords_data[1:]:
            if p[0] < 0 or p[1] < 0:
                break
            if grad:
                color[0] -= 1 / 255
            cv2.line(
                image,
                (int(prev_p[0]), int(prev_p[1])),
                (int(p[0]), int(p[1])),
                color,
                thickness,
            )
            prev_p = p

    elif plot_type == "poly":
        cv2.polylines(
            image,
            np.array([coords_data[:, :2]], dtype=np.int),
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )

    elif plot_type == "point":
        logger.error("plot point mode is not implemented!")
        sys.exit()

    cv2.imwrite(savepath, image * 255)


def is_jupyter_notebook():
    """
    Checking if the running environment is jupyter notebook or not.

    Returns
    -------
    <return> : bool
        True for jupyter detected, False for normal python detected.
    """

    # 1. normal python shell
    if "get_ipython" not in globals():
        return False

    # 2. ipython shell
    try:
        # get_ipython() is available in the global namespace by default when
        # iPython is started.
        if get_ipython().__class__.__name__ == "TerminalInteractiveShell":
            return False

        # 3. jupyter notebook
        else:
            return True
    except NameError:
        return False
