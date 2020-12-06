import os
import re
from pathlib import Path
from typing import Union, Dict
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .yolov3.utils.utils import xywh2xyxy, plot_one_box
from .yolov3.utils.datasets import letterbox
from .planercnn.visualize_utils import unmold_image, draw_instances


def extract_weights_from_model(
    model: torch.nn.Module,
    from_layer: int,
    path: str,
    prefix_keys_with: str = "module_list.",
):
    """
    Extracts weights from a pretrained model, given the layer from which to extract from, and 
    saves the weights.

    Args:
        model (torch.nn.Module): The pretrained model to extract from.
        from_layer (int): The layer to extract from.
        path (str): The path to save the weights.
        prefix_keys_with (str, optional): prefix to add to the weight keys. Defaults to "module_list.".
    """
    modules = list(model.children())[0][from_layer:]
    if prefix_keys_with is not None:
        data = {}
        for key, value in modules.state_dict().items():
            data[f"{prefix_keys_with}{key}"] = value
    else:
        data = modules.state_dict().items()
    torch.save({"model": data}, path)


def extract_weights_from_checkpoint(
    inp: Union[str, Dict], dest: str, regex: str, exclude_matched: bool = True
):
    """
    Extracts keys (eg: weights, etc) from a torch checkpoint using a regex and saves the result.

    Args:
        inp (Union[str, Dict]): The input. Should be a path to the checkpoint file or the checkpoint data itself.
        dest (str): The path to save the result to.
        regex (str): The regex used to filter the keys.
        exclude_matched (bool, optional): Whether the regex should be used to match items to remove \
            (happens when True) or items to include (happens when False). Defaults to True.
    """
    if type(inp) == str:
        inp = torch.load(inp)

    data = {k: v for k, v in inp.items() if exclude_matched ^ bool(re.match(regex, k))}
    torch.save(data, dest)


def parse_data_cfg(path):
    # Parses the data configuration file
    with open(path, "r") as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        key, val = line.split("=")
        val = val.strip()

        if val.startswith("./"):
            val = val.replace("./", str(Path(path).parent) + os.sep)
        elif val.startswith("../"):
            val = val.replace(
                "../", str(Path(os.path.abspath(path)).parent.parent) + os.sep
            )

        options[key.strip()] = val

    return options


def plot_yolo_bbox(
    images, targets, paths=None, names=None, max_size=640, max_subplots=16
):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()["color"]]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = (
                None if gt else image_targets[:, 6]
            )  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = "%s" % cls if gt else "%s %.1f" % (cls, conf[j])
                    plot_one_box(
                        box, mosaic, label=label, color=color, line_thickness=tl
                    )

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        # Image border
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )

    mosaic = cv2.resize(
        mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
    )

    return mosaic


def construct_midas_depth(depth, bits=1):
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        out = out.type(torch.uint8)
    elif bits == 2:
        out = out.type(torch.uint8)

    return out


def visualize_planercnn_batch(config, input_pair, detection_pair):
    _, c, h, w = input_pair[0]["image"].shape
    b = len(input_pair)
    
    res = np.zeros((b, h, w, c))
    for i, (input_dict, detection_dict) in enumerate(zip(input_pair, detection_pair)):
        res[i] = 255 - letterbox(
            visualize_planercnn(config, input_dict, detection_dict),
            (h, w),
            auto=False
        )[0]

    return res


def visualize_planercnn(config, input_dict, detection_dict):
    images = input_dict["image"].detach().cpu().numpy().transpose((0, 2, 3, 1))
    images = unmold_image(images, config)
    image = images[0]

    depths = input_dict["depth"].detach().cpu().numpy()
    depth_gt = depths[0]

    if "detection" in detection_dict and len(detection_dict["detection"]) > 0:
        detections = detection_dict["detection"].detach().cpu().numpy()
        detection_masks = (
            detection_dict["masks"].detach().cpu().numpy().transpose((1, 2, 0))
        )
        if "flag" in detection_dict:
            detection_flags = detection_dict["flag"]
        else:
            detection_flags = {}
            pass
        instance_image, normal_image, depth_image = draw_instances(
            config,
            image,
            depth_gt,
            detections[:, :4],
            detection_masks > 0.5,
            detections[:, 4].astype(np.int32),
            detections[:, 6:],
            detections[:, 5],
            draw_mask=True,
            transform_planes=False,
            detection_flags=detection_flags,
        )
        return instance_image[80:560]
    else:
        return image
