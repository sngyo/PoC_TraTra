import logging
import os

import torch
from tqdm import tqdm

from utils.utils import load_model, plot_output
from data_utils.dataloader import get_dataloader, get_inf_dataloader
from modules.tratra import TraTra


logger = logging.getLogger(__name__)


def run_test(configs):
    """
    called from main.py
        1. run for all test dataset images
        2. specify image name or directory name
    """
    # FIXME: hard-coded save path
    savedir = os.path.join(configs.root_save_dir, "test_results")
    os.makedirs(savedir, exist_ok=True)

    # model load
    model = TraTra(configs).to(configs.device)
    model, _ = load_model(model, None, configs.load_model, configs.device)
    model.eval()

    if configs.test_data_path is None:
        logger.info(
            "configs.test_data_path is None, "
            "so run model on entire test data set. This may take while..."
        )
        _, _, inf_loader = get_dataloader(configs)

    else:
        logger.info(f"run model on images in [{configs.test_data_path}]")
        inf_loader = get_inf_dataloader(configs)

    for imgs, fnames in tqdm(inf_loader):
        imgs = imgs.float().to(configs.device)
        with torch.no_grad():
            h_maps, coords, pt_states = model(imgs)

        # save results
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)

        pred_coords = coords.cpu().numpy()
        pred_pt_states = pt_states.cpu().numpy()

        # NOTE: for now, only mono-stroke
        n_pts = pred_pt_states.argmax(axis=1)

        # TODO: multi-process?
        for i in range(imgs.shape[0]):
            savepath = os.path.join(savedir, os.path.basename(fnames[i]))
            plot_output(
                imgs[i],
                pred_coords[i, :n_pts[i]],
                savepath,
                color=[255/255, 140/255, 255/255],
                grad=True,
            )

    logger.info(f'Results saved at {savedir}')
    logger.info('Test script finished!')
