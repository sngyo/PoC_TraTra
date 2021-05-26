import argparse
import os
import shutil
import sys

import torch
import yaml
from easydict import EasyDict


def yaml_loader(yaml_path, configs, logger):
    """
    Loads .yaml file and update configs.

    Parameters
    ----------
    yaml_path : str
        A path to .yaml.
    configs: EasyDict
    logger : logger

    Returns
    -------
    configs: EasyDict
    """
    with open(yaml_path, "r") as f:
        params_data = yaml.load(f, Loader=yaml.SafeLoader)

    # update arguments
    for k, v in params_data.items():
        configs[k] = v
    return configs


def save_configs(configs, logger):
    """
    Supposed to be called from main.py after all argument updates are done

    Parameters
    ----------
    configs : EasyDict
    logger : logger
    """
    logger.info("=" * 80)
    logger.info(" " * 5 + "Arguments" + " " * 17 + "|" + " " * 5 + "Values")
    logger.info("-" * 80)
    _configs = sorted(configs.items())
    for cf, value in _configs:
        logger.info(f" {cf:30}: {value}")
    logger.info("=" * 80)


def update_configs(configs, logger):
    """
    Default arugments update settings are here.

    Parameters
    ----------
    configs : EasyDict
    logger : logger

    Returns
    -------
    configs : ArgumentParser
    """
    # load yaml file then update configs
    if configs.yaml_path:
        configs = yaml_loader(configs.yaml_path, configs, logger)

    # device check
    if torch.cuda.is_available():
        if configs.multi_gpus:
            logger.warning(
                "Multi GPU mode activated --> ignoring `--gpu_id` argument "
                "and use `configs.multi_gpu_idxs` instead"
            )
            configs.device = "cuda"
        if configs.gpu_id >= torch.cuda.device_count():
            previous_id = configs.gpu_id
            configs.gpu_id = torch.cuda.device_count() - 1
            logger.warning(
                f"gpu id updated to {configs.gpu_id} from {previous_id}"
            )
        configs.device = (
            f"cuda:{configs.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using cuda:{configs.gpu_id} for training.")
    else:
        logger.warning("GPU is not available!")

    # model save path
    save_dir_path = os.path.join(configs.root_save_dir, configs.name)

    skip_check = False
    # -------------------------------------------------------------------------
    # hard-coded
    check_skip_command = [
        configs.notebook_mode,
        configs.test_tratra,
        configs.cvt_xml2unp,
        configs.gen_iam_single_stroke_data,
        configs.gen_iam_resampled_data,
        configs.gen_iam_exclude_filename_data,
    ]
    # -------------------------------------------------------------------------
    for cmd in check_skip_command:
        if cmd:
            skip_check = True
            break

    if skip_check:
        pass
    elif os.path.isdir(save_dir_path) and not configs.overwrite:
        logger.warning(f"exp name [{configs.name}] exist!")
        sys.exit()
    else:
        if os.path.isdir(save_dir_path):
            shutil.rmtree(save_dir_path)
        os.makedirs(save_dir_path)
        configs.save_model = os.path.join(save_dir_path, configs.save_model)
    return configs


def get_configs(logger, args=None):
    """
    Parameters
    ----------
    logger
    args: list, default is None
        such as ['--batchsize', '200', '--epoch', '100', '-g']

    Returns
    -------
    configs : EasyDict()
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trajectory Extraction by Transformer official model",
    )

    # ==============================
    # Configurations
    # ==============================
    # yaml setting file
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=None,
        help="A path to .yaml file which stores training parameter settings.",
    )

    parser.add_argument(
        "--gpu_id", type=int, default=0, help="gpu id (0 ~ max_gpu_num-1)"
    )
    parser.add_argument(
        "--notebook_mode",
        action="store_true",
        help="notebook_mode (to skip directory check)",
    )

    # ==============================
    # Scripts
    # ==============================
    parser.add_argument(
        "--train_tratra",
        action="store_true",
        help="training TraTra model (single stroke for now)",
    )
    parser.add_argument(
        "--test_tratra",
        action="store_true",
        help="test TraTra model (single stroke for now)",
    )

    # iamonline
    parser.add_argument(
        "--cvt_xml2unp",
        action="store_true",
        help="Converting xml files to unp files",
    )
    parser.add_argument(
        "--gen_iam_single_stroke_data",
        action="store_true",
        help="Generate new IAM dataset only containing single stroke data",
    )
    parser.add_argument(
        "--gen_iam_resampled_data",
        action="store_true",
        help="Generate resampled IAM dataset unp data",
    )
    parser.add_argument(
        "--gen_iam_exclude_filename_data",
        action="store_true",
        help="Generate exclude_...txt containing too much point in a file",
    )

    # 1. parsing
    if args is not None:
        args = parser.parse_args(args=args)
    else:
        args = parser.parse_args()

    # 2. convert to EasyDict()
    configs = EasyDict(args.__dict__)

    # 3. update arguments
    configs = update_configs(configs, logger)

    return configs
