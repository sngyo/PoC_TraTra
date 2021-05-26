import logging
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from skimage.util import random_noise
from tqdm import tqdm

import data_utils.data_setup as ds
from utils.utils import get_specified_ext_fnames

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def load_org_xml(xml_path):
    """
    loading original IAMONLINE dataset's xml file and extracting strokes info

    Parameters
    ----------
    xml_path : str
    """
    with open(xml_path) as f:
        try:
            xml_string = f.read()
        except UnicodeDecodeError:
            logger.error(f"UnicodeDecodeError: {xml_path}")
            pass

    root = ET.fromstring(xml_string)
    strokes = []
    for stroke in root.iter("Stroke"):
        fmt_stroke = []
        for pt in stroke:
            fmt_stroke.append([pt.attrib["x"], pt.attrib["y"]])
        strokes.append(fmt_stroke)
    return strokes


# -----------------------------------------------------------------------------
# execution scripts (should be called from main.py)
# -----------------------------------------------------------------------------
def convert_xml_to_unp(xml_root_path, unp_root_path):
    """
    converting xml files into unp files (sngyo version)
    """
    # check xml_root_path
    if "lineStrokes" not in xml_root_path:
        logger.error(
            "Because of hard-coded path config, you should use `lineStrokes` "
            "directory for `xml_root_path` for now"
        )

    # start!
    xml_names = get_specified_ext_fnames(xml_root_path, ".xml")
    logger.info(f"{len(xml_names)} files found!")

    all_strokes = {}
    for fname in tqdm(xml_names):
        all_strokes[fname] = load_org_xml(fname)

    for fname in tqdm(xml_names):
        # ----------------
        # HARD-coded here!
        savename = fname.replace(xml_root_path, unp_root_path).replace('.xml', '.unp')
        # ----------------

        os.makedirs(os.path.dirname(savename), exist_ok=True)
        ds.save_sngyo_unp(all_strokes[fname], savename)

    logger.info("IAMONLINE's xml files are converted to unp files")


def gen_iam_single_stroke_data(
        unp_root_path, img_h, img_w, img_pad, gen_iam_path
):
    """
    generate image and unp file for mono-stroke data

    Parameters
    ----------
    unp_path : str
        this must be org_unp directory
    img_h : int
    img_w : int
    img_pad : int
        padding pixel size
    gen_iam_path : str
        root directory for new data
        gen_iam_path
            |-unp
            |-fig
    """
    # 1. load original unp data
    unp_names = get_specified_ext_fnames(unp_root_path, ".unp")

    # 2. for each unp file, extracting and saving stroke file
    stroke_data = {}

    # 2-1. load data
    for fname in unp_names:
        strokes = ds.load_sngyo_unp(fname)
        for i_s, stroke in enumerate(strokes):
            # FIXME: this is not elegant...
            sname = fname.replace(".unp", f"{i_s}.unp")
            stroke_data[sname] = stroke
    logger.info(f"{len(stroke_data)} strokes exists")

    # 2-2. move coords towards (0,0) and discard oversize strokes
    # 2-3. generate image
    discard_ls = []
    for sname, stroke in tqdm(stroke_data.items()):
        # [[x1, y1], [x2, y2], ...] --> [[x1, x2, ...], [y1, y2, ...]]
        tstroke = np.array(stroke).transpose()
        x_max = tstroke[0].max()
        x_min = tstroke[0].min()
        y_max = tstroke[1].max()
        y_min = tstroke[1].min()

        # oversize
        # TODO: resize ? for now, discarding oversize strokes
        if (x_max - x_min > img_w - 2 * img_pad) or \
           (y_max - y_min > img_h - 2 * img_pad):
            discard_ls.append(sname)
            continue

        # update x (width)
        tstroke[0] += int(img_w / 2 - (x_max + x_min) / 2)
        # update y (height)
        tstroke[1] += int(img_h / 2 - (y_max + y_min) / 2)

        # NOTE: max_p (too long) will be considered after resampling

        new_stroke = np.array([tstroke.transpose()])

        # save new unp file
        # FIXME: hard-coded
        savename = sname.replace("org_unp", f"gen_iam_{img_h}/unp")
        if img_h != img_w:
            logger.warning(f"Directory name maybe confusing... : {savename}")
        ds.save_sngyo_unp(new_stroke, savename)

        # generate image
        gen_img = ds.generate_img(new_stroke, img_h, img_w, thickness=2)
        gen_img = random_noise(gen_img / 255.0) * 255.0
        gen_img = ds.apply_blur(gen_img)

        # FIXME: dangerous, hard-coded
        figname = savename.replace(".unp", ".png").replace("unp", "fig")
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        cv2.imwrite(figname, gen_img)

    logger.info(f"{len(discard_ls)} strokes are dicarded")
    logger.info(
        f"Therefore, we have {len(stroke_data) - len(discard_ls)} strokes"
    )
