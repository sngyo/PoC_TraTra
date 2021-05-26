import dsntnn
import torch
import torch.nn as nn


def custom_loss(
    pred_heatmaps,
    pred_coords,
    pred_pt_states,
    targets,
    device,
    in_img_size=(128, 128),
):
    """
    Parameters
    ----------
    pred_heatmaps : (batch_size, embed_dim, h, w)
    pred_coords :
    pred_pt_states :
    targets: (batch_size, embed_dim, 3)
        [[x0, y0, 0], [x1, y2, 0],,, [xn, yn, 1], [-1, -1, -1],,,]
    device :
    in_img_size :
    """
    batch_size = targets.shape[0]
    embed_dim = targets.shape[1]
    in_img_size = torch.Tensor(in_img_size).to(device)

    # [1, 1, 1, 0, 0, 0] (padding part will be 0)
    mask_target_penstate = 1 - (targets[:, :, 2] < 0).to(torch.int)
    target_mask = mask_target_penstate.unsqueeze(-1).float()
    target_coords = targets[:, :, :2] * target_mask
    normalized_target_coords = (target_coords) * 2 + 1 / in_img_size - 1
    pred_coords = pred_coords * target_mask

    # the number of validated points
    nb_pts = mask_target_penstate.sum(dim=1)

    # # 1. coords mse
    # coords_mse_loss = 0
    # for i in range(batch_size):
    #     coords_mse_loss += nn.MSELoss()(
    #         pred_coords[i].float(), target_coords[i].float()
    #     ) * embed_dim / nb_pts[i]
    # coords_mse_loss /= batch_size

    # 2. L1 loss
    coords_l1_loss = 0
    for i in range(batch_size):
        coords_l1_loss += (
            nn.L1Loss()(pred_coords[i].float(), target_coords[i].float())
            * embed_dim
            / nb_pts[i]
        )
    coords_l1_loss /= batch_size

    # 3. Jensen-shannon divergence loss
    # What is JS-divergence ?
    # --> When there are two probability distributions, KL-divergence is
    #     well known as a method to measure these distances, but a symmetric
    #     version of KL-divergence is JS-divergence
    reg_loss = 0
    for i, element in enumerate(pred_heatmaps):  # batch loop
        reg_loss += (
            dsntnn.js_reg_losses(
                element[: nb_pts[i]].unsqueeze(1),
                (normalized_target_coords[i, : nb_pts[i]]).unsqueeze(1),
                sigma_t=1.0,
            ).sum()
            / nb_pts[i]
        )
    reg_loss /= batch_size

    # 4.pen_state_loss
    pen_state_loss = nn.BCELoss()(
        pred_pt_states, (targets[:, :, 2] == 1).to(torch.int).float()
    )

    return reg_loss, coords_l1_loss, pen_state_loss
