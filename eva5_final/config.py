from typing import Tuple

import os
from dataclasses import dataclass

from .planercnn.config import Config as _Config


@dataclass
class YoloV3Options:
    data: str = "data/yolo/custom.data"
    multi_scale: bool = False  # adjust (67% - 150%) img_size every 10 batches
    epochs: int = 300
    batch_size: int = 16
    cfg: str = "config/yolov3-spp-detector.cfg"
    img_size: Tuple[int, int, int] = (320, 640, 640)  # [min_train, max-train, test]
    rect: bool = False  # rectangular training
    resume: bool = False  # resume training from last.pt
    nosave: bool = False  # only save final checkpoint
    notest: bool = False  # only test final epoch
    evolve: bool = False  # evolve hyperparameters
    bucket: str = ""  # gsutil bucket
    cache_images: bool = False  # cache images for faster training
    weights: str = "weights/yolo-detector.pt"
    name: str = ""  # renames results.txt to results_name.txt if supplied
    device: str = ""  # device id (i.e. 0 or 0,1 or cpu)
    adam: bool = False  # use adam optimizer
    single_cls: bool = False  # train as single-class dataset
    freeze_layers: bool = False  # Freeze non-output layers
    gr: float = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    conf_thres: float = 0.001
    iou_thres: float = 0.6  # for nms
    mosiac: bool = False  # apply recap kind of augmentation


@dataclass
class PlaneRCNNOptions:
    numEpochs: int
    batchSize: int
    gpu: int = 1
    task: str = "train"  # [train, test, predict]
    anchorFolder: str = "eva5_final/planercnn/anchors"
    LR: float = 1e-5
    heatmapThreshold: float = 0.5
    distanceThreshold3D: float = 0.2
    distanceThreshold2D: float = 20.0
    width: int = 640
    height: int = 512
    suffix: str = "warping_refine"
    maskWidth: int = 56
    maskHeight: int = 56
    anchorType: str = "normal"
    numAnchorPlanes: int = 0
    frameGap: int = 20
    planeAreaThreshold: int = 500  # probably not used
    planeWidthThreshold: int = 10  # probably not used
    scaleMode: str = "variant"
    cornerPositiveWeight: int = 0
    positiveWeight: float = 0.33
    maskWeight: int = 1
    warpingWeight: float = 0.1
    convType: str = "2"
    dataset: str = '' # used in planercnn's config


class GlobalConfig:
    EPOCHS = 100
    BATCH_SIZE = 16
    DATA_DIR = "data"
    IMG_SIZE = 64
    MIN_IMG_SIZE = 64  # used in yolo


class MidasConfig:
    WEIGHTS_PATH = "weights/midas.pt"
    FREEZE = True


class YoloV3Config:
    HEAD_CONFIG_PATH = "config/yolov3-head.cfg"
    DETECTOR_CONFIG_PATH = "config/yolov3-spp-detector.cfg"
    DETECTOR_WEIGHTS_PATH = "weights/yolo-detector.pt"

    hyp = {
        "giou": 3.54,  # giou loss gain
        "cls": 37.4,  # cls loss gain
        "cls_pw": 1.0,  # cls BCELoss positive_weight
        "obj": 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
        "obj_pw": 1.0,  # obj BCELoss positive_weight
        "iou_t": 0.20,  # iou training threshold
        "lr0": 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
        "lrf": 0.0005,  # final learning rate (with cos scheduler)
        "momentum": 0.937,  # SGD momentum
        "weight_decay": 0.0005,  # optimizer weight decay
        "fl_gamma": 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
        "degrees": 1.98 * 0,  # image rotation (+/- deg)
        "translate": 0.05 * 0,  # image translation (+/- fraction)
        "scale": 0.05 * 0,  # image scale (+/- gain)
        "shear": 0.641 * 0,  # image shear (+/- deg)
    }

    opt = YoloV3Options(
        os.path.join(GlobalConfig.DATA_DIR, "yolo", "custom.data"),
        epochs=GlobalConfig.EPOCHS,
        img_size=(
            GlobalConfig.MIN_IMG_SIZE,
            GlobalConfig.IMG_SIZE,
            GlobalConfig.IMG_SIZE,
        ),
    )


class PlaneRCNNConfig(_Config):
    EXTERNAL_EXTRACTOR = True
    NUM_CLASSES = 4
    GPU_COUNT = 1
    INIT_LOG_DIR_AND_WEIGHTS = False # preventing planercnn model from making a log directory and loading weights
    MODEL_WEIGHTS_PATH = "weights/planercnn/checkpoint-partial.pth"
    REFINE_MODEL_WEIGHTS_PATH = "weights/planercnn/checkpoint-refine.pth"
    options = PlaneRCNNOptions(
        numEpochs=GlobalConfig.EPOCHS, batchSize=GlobalConfig.BATCH_SIZE
    )

    def __init__(self):
        super(PlaneRCNNConfig, self).__init__(PlaneRCNNConfig.options)


class Config(GlobalConfig):
    yolo_config = YoloV3Config
    midas_config = MidasConfig
    planercnn_config = PlaneRCNNConfig()
