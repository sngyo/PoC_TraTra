import logging
import os
import tempfile
from statistics import mean

import torch
import wandb
from data_utils.dataloader import get_dataloader
from fastprogress import master_bar, progress_bar
from modules.tratra import TraTra
from torch.optim.lr_scheduler import StepLR
from utils.loss import custom_loss
from utils.utils import load_model, plot_output, seed_everything

logger = logging.getLogger(__name__)


class AverageCustomLossMeter:
    """
    Computing and storing the average and the current loss value
    """

    def __init__(self, n_losses):
        """
        Parameters
        ----------
        n_losses : int
            The number of losses
        """
        self.n_losses = n_losses
        self.reset()

    def reset(self):
        self._avg = torch.zeros(self.n_losses)
        self._running_total_losses = torch.zeros(self.n_losses)
        self._count = 0

    def update(self, curr_batch_losses, batch_size):
        """
        Parameters
        ----------
        curr_batch_losses : torch.tensor of float
            torch.Tensor([loss1, loss2, loss3])
        batch_size : int
        """
        self._running_total_losses += curr_batch_losses * batch_size
        self._count += batch_size
        self._avg = self._running_total_losses / self._count

    def get_avg_loss(self):
        """
        Returns
        -------
        self._avg : numpy.array
            numpy.array([avg_l1, avg_l2, ...])
        """
        return self._avg.numpy()


def train_tratra(configs):
    seed_everything(configs.seed)
    device = configs.device
    logger.info(f"Detected device type: {device}")

    # wandb
    if not configs.off_wandb:
        wandb.init(project=configs.wdb_proj, name=configs.name)
        wandb.config.update(configs)
        wandb.save(configs.yaml_path)

    # dataloader
    train_loader, valid_loader, test_loader = get_dataloader(configs)

    # modeling
    model = TraTra(configs).to(device)
    if configs.multi_gpus:
        logger.warning("Requiring debug before using multi gpus mode")
        model = torch.nn.DataParallel(model, device_ids=configs.multi_gpu_idxs)

    # TODO: SAM
    if configs.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    else:
        logger.error(f"{configs.optimizer} is not implemented yet")
    scheduler = StepLR(
        optimizer, step_size=configs.lr_stepsize, gamma=configs.lr_gamma
    )
    criterion = custom_loss

    # params
    init_epoch = 1
    best_loss = 1e7
    early_stop_count = 0
    summary_train_loss = AverageCustomLossMeter(configs.n_losses)
    summary_valid_loss = AverageCustomLossMeter(configs.n_losses)
    summary_test_loss = AverageCustomLossMeter(configs.n_losses)

    # load pre-train weight
    if configs.load_model is not None:
        model, optimizer = load_model(
            model, optimizer, configs.load_model, device
        )
        init_epoch = model.info_dict["next_epoch"]
        best_loss = model.info_dict["best_loss"]
        logger.info(f"model loaded (path: {configs.load_model})")
    else:
        logger.info("Model initialized successfully")

    mb = master_bar(range(init_epoch, configs.epochs + 1))
    for epoch in mb:
        # --------------------
        # Training loop
        # --------------------
        model.train()
        for x, y, _ in progress_bar(train_loader, parent=mb):
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            h_maps, coords, pt_state = model(x)
            r_l, c_l, p_l = criterion(h_maps, coords, pt_state, y, device)
            loss_train = r_l + c_l + p_l  # TODO: coef.
            loss_train.backward()
            optimizer.step()
            summary_train_loss.update(
                torch.tensor([r_l.item(), c_l.item(), p_l.item()]), x.shape[0]
            )
            if configs.only_first_batch:
                break

        # --------------------
        # valid loop
        # --------------------
        model.eval()
        for x, y, _ in progress_bar(valid_loader, parent=mb):
            x = x.float().to(device)
            y = y.float().to(device)

            with torch.no_grad():
                h_maps, coords, pt_state = model(x)
                r_l, c_l, p_l = criterion(h_maps, coords, pt_state, y, device)
                summary_valid_loss.update(
                    torch.tensor([r_l.item(), c_l.item(), p_l.item()]),
                    x.shape[0],
                )
            if configs.only_first_batch:
                break

        # --------------------
        # Test loop
        # --------------------
        model.eval()
        for x, y, test_fnames in progress_bar(test_loader, parent=mb):
            x = x.float().to(device)
            y = y.float().to(device)

            with torch.no_grad():
                h_maps, coords, pt_state = model(x)
                r_l, c_l, p_l = criterion(h_maps, coords, pt_state, y, device)
                summary_test_loss.update(
                    torch.tensor([r_l.item(), c_l.item(), p_l.item()]),
                    x.shape[0],
                )
            if configs.only_first_batch:
                break

        # --------------------
        # Save model
        # --------------------
        mean_train_losses = mean(summary_train_loss.get_avg_loss())
        mean_valid_losses = mean(summary_valid_loss.get_avg_loss())
        mean_test_losses = mean(summary_test_loss.get_avg_loss())

        if mean_valid_losses < best_loss:
            # update model
            best_loss = mean_valid_losses
            early_stop_count = 0
            save_dict = {
                "next_epoch": epoch + 1,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            }
            model.info_dict = save_dict
            torch.save(model, configs.save_model)
        else:
            early_stop_count += 1
            if early_stop_count == configs.patience:
                logger.info(f"Early Stopping: epoch {epoch}")
                break

        curr_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        msg = (
            "epoch: {}/{} - lr: {:.6f} - train: {:.5f} - val: {:.5f} - test: {:.5f}"
        ).format(
            epoch,
            configs.epochs,
            curr_lr,
            mean_train_losses,
            mean_valid_losses,
            mean_test_losses,
        )
        logger.info(msg)

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        # --------------------
        # wandb logging
        # --------------------
        if not configs.off_wandb:
            logging_info_dict = {
                "learning_rate": curr_lr,
                "train_loss": mean_train_losses,
                "train_reg_loss": summary_train_loss.get_avg_loss()[0],
                "train_coords_l1_loss": summary_train_loss.get_avg_loss()[1],
                "train_pen_state_loss": summary_train_loss.get_avg_loss()[2],
                "valid_loss": mean_valid_losses,
                "valid_reg_loss": summary_valid_loss.get_avg_loss()[0],
                "valid_coords_l1_loss": summary_valid_loss.get_avg_loss()[1],
                "valid_pen_state_loss": summary_valid_loss.get_avg_loss()[2],
                "test_loss": mean_test_losses,
                "test_reg_loss": summary_test_loss.get_avg_loss()[0],
                "test_coords_l1_loss": summary_test_loss.get_avg_loss()[1],
                "test_pen_state_loss": summary_test_loss.get_avg_loss()[2],
            }
            wandb.log(logging_info_dict)

            # plot the output on the original input image
            src_imgs = x.cpu().numpy().transpose(0, 2, 3, 1)
            targets = y.cpu().numpy()
            n_pts = (targets[:, :, 2] >= 0).sum(axis=1)
            pred_coords = coords.cpu().numpy()

            with tempfile.TemporaryDirectory() as dname:
                for i, fname in enumerate(test_fnames):
                    fname = (
                        f"{epoch}_"
                        + os.path.splitext(os.path.basename(fname))[0]
                        + ".png"
                    )
                    save_file = os.path.join(dname, fname)
                    plot_output(
                        src_imgs[i],
                        pred_coords[i, :n_pts[i]],
                        save_file,
                    )
                    wandb.log({fname: wandb.Image(save_file)})

        # reset summary meter
        summary_train_loss.reset()
        summary_valid_loss.reset()
        summary_test_loss.reset()
