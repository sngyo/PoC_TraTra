import logging
import os
import random
import zipfile

import cv2
import gdown
import numpy as np
from tqdm import tqdm

from utils.utils import check_file_exist, get_specified_ext_fnames


# Gdrive dataset path (hard-coded)
DATASETS = [
    ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "./data/iam_online/iam_online.zip"),
]

logger = logging.getLogger(__name__)


def download_unzip_datasets():
    """
    You can use this function to download
    IAMONLINE dataset from your google drive
    GID should be hard-coded above (replace xxxx with the GID)
    """
    # Prepare Directories
    os.makedirs("./data/iam_online", exist_ok=True)

    for did, path in DATASETS:
        if os.path.isfile(path):
            continue

        logger.info(f"Download to {path}")
        url = f"https://drive.google.com/uc?id={did}"
        gdown.download(url, path, quiet=False)

    # unzip data
    logger.info("Unzip download file...")
    with zipfile.ZipFile("./data/iam_online/iam_online.zip") as existing_zip:
        existing_zip.extractall("./data/iam_online/")
        logger.info("IAM_Online done !")


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def load_sngyo_unp(fname):
    """
    loading generated unp file (sngyo version unp)

    Example
    ```
    X Y
    72.0 58.0
    73.0 57.0
    74.0 55.0
    =====
    32.0 38.0
    33.0 47.0
    34.0 45.0
    =====
    ```

    Parameters
    ----------
    fname: string
    """
    check_file_exist(fname)
    strokes = []

    with open(fname, "r", encoding="latin-1") as f:
        lines = f.readlines()
        stroke = []

        # skip first line (X, Y) information
        for line in lines[1:]:
            if "=====" in line:
                strokes.append(stroke)
                stroke = []
            else:
                coord = line.split()
                stroke.append([float(coord[0]), float(coord[1])])
    return strokes


def save_sngyo_unp(coords, fpath):
    """
    save coords data as unp file

    Parameters
    ----------
    coords : list of list of tuple
        strokes -> stroke -> coordinates
    fname : string
    """
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    with open(fpath, mode="w", encoding="latin-1") as f:
        f.write("X Y\n")
        for stroke in coords:
            for p in stroke:
                f.write(f"{p[0]} {p[1]}\n")
            f.write("=====\n")  # sign of end of a stroke


def generate_img(coords, img_h, img_w, poly=False, color=(1), thickness=1):
    """
    Generate character image from coordinates data

    Parameters
    ----------
    coords : list of list of tuple
    img_h : int
    img_w : int
    poly : bool, default is False
    color : tuple, default is (1)
    thickness : int, default is 1
    """
    img = np.zeros((img_h, img_w))

    for stroke in coords:
        if not poly:
            # the first coordinates in a stroke is PEN_DOWN point
            prev_p = stroke[0][0], stroke[0][1]
            for p in stroke[1:]:
                cv2.line(
                    img,
                    (int(prev_p[0]), int(prev_p[1])),
                    (int(p[0]), int(p[1])),
                    color,
                    thickness,
                )
                prev_p = p
        else:
            # polyline model
            color = random.randint(200, 255)
            thickness = random.randint(2, 3)
            cv2.polylines(
                img,
                np.array([stroke], dtype=np.int),
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return ((1 - img) * 255).astype(np.int)


def add_imp_noise(src, max_num_dot=500):
    """
    Noise function to make generated images realistic
    """
    h, w = src.shape[0], src.shape[1]

    p_noise = random.uniform(0.3, 1.0)
    p_x = np.random.randint(0, w - 1, int(max_num_dot * p_noise))
    p_y = np.random.randint(0, h - 1, int(max_num_dot * p_noise))

    # DEBUG check if it's okay for RGB image
    src[p_y, p_x] = np.random.randint(100, 255)
    return src


def add_gaussian_noise(src, mean=0, var=0.1, sigma=15):
    gauss = np.random.normal(mean, sigma, src.shape)
    noisy = src + gauss
    return noisy


def apply_blur(src, k_size=5, sigma=1.5):
    blur_img = cv2.GaussianBlur(src, (k_size, k_size), sigma)
    return blur_img


def get_subdivided_coords(start, end, alpha=0.01):
    """
    Fill in the blank(?) between two points(stard & end) with distance alpha

    Parameters
    ----------
    start: [x, y]  (np.float)
    end: [x, y]  (np.float)
    alpha: distance for subdivision (default: 0.01)

    Returns
    -------
    ndarray of points(ndarray) start from `start`, not including `end`
    """
    if start[0] == end[0] and start[1] == end[1]:
        return np.array([start])

    norm = np.linalg.norm(end - start)

    if start[0] == end[0]:  # same x position
        dy = (end[1] - start[1]) * alpha / norm
        y = np.arange(start[1], end[1], dy)
        x = np.ones_like(y) * start[0]
        return np.array([x, y]).transpose(1, 0)

    if start[1] == end[1]:  # same y position
        dx = (end[0] - start[0]) * alpha / norm
        x = np.arange(start[0], end[0], dx)
        y = np.ones_like(x) * start[1]
        return np.array([x, y]).transpose(1, 0)

    dx = (end[0] - start[0]) * alpha / norm
    dy = (end[1] - start[1]) * alpha / norm
    x = np.arange(start[0], end[0], dx)
    y = np.arange(start[1], end[1], dy)

    if len(x) != len(y):
        if np.abs(len(x) - len(y)) > 1:
            logger.error(
                f"get_subdivided_coords - len(x): {len(x)} - len(y): {len(y)}"
            )
        else:
            # abs(len(x) - len(y)) = 0 or 1
            min_len = np.minimum(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

    return np.array([x, y]).transpose(1, 0)


# -----------------------------------------------------------------------------
# execution scripts (should be called from main.py)
# -----------------------------------------------------------------------------
def resampling_unp(
    unp_dir_path, gen_dir_path, alpha=0.01, resample_dist=3, max_p=256
):
    """
    Parameters
    ----------
    unp_dir_path : str
        loading path
    gen_dir_path : str
        saving root path
    alpha : float, default is 0.01
    resample_dist : float, default is 3
    max_p : int, default is 256
        after resampling, if a file contains more than max_p points,
        these filenames will be stored in exclude file
    """
    unp_fnames = get_specified_ext_fnames(
        os.path.join(unp_dir_path, "unp"), ".unp"
    )
    logger.info(f"{len(unp_fnames)} unp files found!")

    max_len_coords = 0
    unp_data = {}

    for fname in tqdm(unp_fnames):
        coords = load_sngyo_unp(fname)
        unp_data[fname] = coords
        max_len_coords = np.maximum(max_len_coords, len(coords[0]))
    logger.info(f"Max length of coordinates: {max_len_coords}")

    step = int(resample_dist / alpha)

    for fname, data in tqdm(unp_data.items()):
        # 1. fill up the gap by 0.01
        new_data = []
        for stroke in data:
            stroke = np.array(stroke)
            new_stroke = []
            for i in range(len(stroke) - 1):
                sub_coords = get_subdivided_coords(
                    stroke[i], stroke[i + 1], alpha=alpha
                )
                new_stroke.extend(sub_coords.tolist())
            new_stroke.append(stroke[-1].tolist())
            new_data.append(new_stroke)

        # 2. resampling
        res_data = []
        for stroke in new_data:
            new_stroke = stroke[::step]

            x = np.linalg.norm(np.array(new_stroke[-1]) - np.array(stroke[-1]))
            if x > resample_dist / 3:
                new_stroke.append(stroke[-1])
            res_data.append(new_stroke)

        # 3. save new file (ex. /resampled_dist3_unp)
        # FIXME: hard-coded
        save_dir_name = os.path.dirname(fname).replace(
            "unp", f"resampled_dist{resample_dist}_unp"
        )
        save_unp_path = os.path.join(save_dir_name, os.path.basename(fname))
        os.makedirs(save_dir_name, exist_ok=True)
        save_sngyo_unp(res_data, save_unp_path)

    # -------------------------------------------------------------------------
    generate_exclude_filenamelist(gen_dir_path, max_p, resample_dist)


def generate_exclude_filenamelist(gen_dir_path, max_p, resample_dist):
    resampled_unp_fnames = get_specified_ext_fnames(
        os.path.join(gen_dir_path, f"resampled_dist{resample_dist}_unp"),
        ".unp",
    )
    max_len_coords = 0
    more_max_p = 0
    more_max_p_fname = []

    for fname in tqdm(resampled_unp_fnames):
        coords = load_sngyo_unp(fname)

        for i_stroke in range(len(coords)):
            max_len_coords = np.maximum(max_len_coords, len(coords[i_stroke]))

            if len(coords[i_stroke]) > max_p:
                more_max_p += 1
                more_max_p_fname.append(fname)
                break

    logger.info(f"# files: {len(resampled_unp_fnames)}")
    logger.info(f"Max length of coordinates: {max_len_coords}")
    logger.info(f"# files containing more than {max_p} points: {more_max_p}")

    # logging filenames that has too many points in order to exclude
    with open(
        os.path.join(
            gen_dir_path, f"exclude_single_resampled_dist{resample_dist}.txt"
        ),
        "w",
    ) as f:
        f.write(f"# contaning more than {max_p} points\n")
        for fname in more_max_p_fname:
            f.write(f"{fname}\n")

    logger.info("exclusive files are registered!")
