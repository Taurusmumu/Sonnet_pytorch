import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss
from .loss import TypeFocalLoss, ForeFocalLoss, BinaryDiceLoss, OrdinalFocalLoss

from collections import OrderedDict


####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    loss_func_dict = {
        "type_focal": TypeFocalLoss(),
        "fore_focal": ForeFocalLoss(),
        "binary_dice": BinaryDiceLoss(),
        "ordinal_focal": OrdinalFocalLoss(),
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    result_dict["nr_type"] = len(run_info["net"]["extra_info"]["weights"]["nt"][run_info["net"]["dataset_name"]])
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    ####
    imgs = batch_data["img"]
    true_np = batch_data["nf_map"]
    true_no = batch_data["no_map"]

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_no = true_no.to("cuda").type(torch.float32)

    # true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "nf": true_np,
        "no": true_no,
    }

    true_tp = batch_data["tp_map"]
    true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
    # true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nt_class_num)
    # true_tp_onehot = true_tp_onehot.type(torch.float32)
    true_dict["nt"] = true_tp

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    pred_dict = model(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] if len(v.shape) == 4 else [k, v.permute(0, 3, 4, 2, 1).contiguous()] for k, v in pred_dict.items()]
    )
    no_predictions = F.softmax(pred_dict["no"], dim=-1)[..., 1]
    no_predictions = torch.sum((no_predictions > 0.5), dim=-1)
    # no_predictions[no_predictions < 0.5] = 0
    # no_predictions[no_predictions >= 0.5] = 1
    # no_predictions = torch.argmin(no_predictions, dim=-1, keepdim=True)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    self_sp = run_info["net"]["extra_info"]["self_sp"]
    weights = run_info["net"]["extra_info"]["weights"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [pred_dict[branch_name], true_dict[branch_name], weights[branch_name][run_info["net"]["dataset_name"]]]
            if self_sp and state_info['epoch'] >= 25:
                loss_args.append(no_predictions)
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict["nf"] = F.softmax(pred_dict["nf"], dim=-1)[..., 1]  # return pos only
    pred_dict["no"] = no_predictions
    pred_dict["nt"] = F.softmax(pred_dict["nt"], dim=-1)
    pred_dict["nt"] = torch.argmax(pred_dict["nt"], dim=-1, keepdim=False)
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict["nf"] = true_np
    true_dict["nt"] = true_tp
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        "nt": (true_dict["nt"], pred_dict["nt"]),
        "nf": (true_dict["nf"], pred_dict["nf"]),
        "no": (true_dict["no"], pred_dict["no"])
    }
    return result_dict


####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["nf_map"]
    true_no = batch_data["no_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).type(torch.int64)
    true_no = torch.squeeze(true_no).type(torch.float32)

    true_dict = {
        "nf": true_np,
        "no": true_no,
    }

    true_tp = batch_data["tp_map"]
    true_tp = torch.squeeze(true_tp).type(torch.int64)
    true_dict["nt"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] if len(v.shape) == 4 else [k,v.permute(0, 3, 4, 2, 1).contiguous()]
             for k, v in pred_dict.items()]
        )
        no_predictions = F.softmax(pred_dict["no"], dim=-1)[..., 1]
        no_predictions = torch.sum((no_predictions > 0.5), dim=-1)
        # no_predictions[no_predictions < 0.5] = 0
        # no_predictions[no_predictions >= 0.5] = 1
        # no_predictions = torch.argmin(no_predictions, dim=-1, keepdim=True)

        pred_dict["nf"] = F.softmax(pred_dict["nf"], dim=-1)[..., 1]
        pred_dict["no"] = no_predictions

        pred_dict["nt"] = F.softmax(pred_dict["nt"], dim=-1)
        pred_dict["nt"] = torch.argmax(pred_dict["nt"], dim=-1, keepdim=False)

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_nf": true_dict["nf"].numpy(),
            "true_no": true_dict["no"].numpy(),

            "pred_nf": pred_dict["nf"].cpu().numpy(),
            "pred_no": pred_dict["no"].cpu().numpy(),
        }
    }
    # if model.module.nr_types is not None:
    result_dict["raw"]["true_nt"] = true_dict["nt"].numpy()
    result_dict["raw"]["pred_nt"] = pred_dict["nt"].cpu().numpy()
    return result_dict


####
def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] if len(v.shape) == 4 else [k,v.permute(0, 3, 4, 2, 1).contiguous()]
             for k, v in pred_dict.items()]
        )
        pred_dict["nf"] = F.softmax(pred_dict["nf"], dim=-1)[..., 1:]
        pred_dict["no"] = F.softmax(pred_dict["no"], dim=-1)[..., 1]
        pred_dict["no"] = torch.sum((pred_dict["no"] > 0.5), dim=-1, keepdim=True)
        # pred_dict["no"][pred_dict["no"] < 0.5] = 0
        # pred_dict["no"][pred_dict["no"] >= 0.5] = 1
        # pred_dict["no"] = torch.argmin(pred_dict["no"], dim=-1, keepdim=True)

        if "nt" in pred_dict:
            type_map = F.softmax(pred_dict["nt"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["nt"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.cpu().numpy()


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    true_np, pred_np = raw_data["nf"]
    true_no, pred_no = raw_data["no"]
    true_tp, pred_tp = raw_data["nt"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_no[idx], 0, 8))
        # if nr_types is not None:  # TODO: a way to pass through external info
        true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_no[idx], 0, 8))
        # if nr_types is not None:
        pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["pred_nf"]
    true_np = raw_data["true_nf"]
    for idx in range(len(raw_data["true_nf"])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value("np_acc", acc_np, "scalar")
    track_value("np_dice", dice_np, "scalar")

    # * TP statistic
    if nr_types is not None:
        pred_tp = raw_data["pred_nt"]
        true_tp = raw_data["true_nt"]
        for type_id in range(0, nr_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_nf"])):
                patch_pred_tp = pred_tp[idx]
                patch_true_tp = true_tp[idx]
                inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                over_inter += inter
                over_total += total
            dice_tp = 2 * over_inter / (over_total + 1.0e-8)
            track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # * HV regression statistic
    pred_no = raw_data["pred_no"]
    true_no = raw_data["true_no"]

    over_squared_error = 0
    for idx in range(len(raw_data["true_nf"])):
        patch_pred_no = pred_no[idx]
        patch_true_no = true_no[idx]
        squared_error = patch_pred_no - patch_true_no
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value("no_mse", mse, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_no = np.array([true_no[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_no = np.array([pred_no[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs, "nf": (true_np, prob_np), "no": (true_no, pred_no)}

    if nr_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["nt"] = (true_tp, pred_tp)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict
