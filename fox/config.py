from typing import Tuple

import os

from .planercnn.config import Config as _Config


def update_attrs(obj, kwargs):
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


class YoloV3Options:
    data: str
    epochs: int
    batch_size: int
    img_size: Tuple[int, int, int]  # [min_train, max-train, test]
    multi_scale: bool = False  # adjust (67% - 150%) img_size every 10 batches
    cfg: str = "config/yolov3-spp-detector.cfg"
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

    def __init__(self, epochs, batch_size, data, img_size_min = 320, img_size_max = 640, img_size_test = 640, **kwargs):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = (img_size_min, img_size_max, img_size_test)

        update_attrs(self, kwargs)


class YoloHyp:
    giou = 3.54  # giou loss gain
    cls = 37.4  # cls loss gain
    cls_pw = 1.0  # cls BCELoss positive_weight
    obj = 64.3  # obj loss gain (*=img_size/320 if img_size != 320)
    obj_pw = 1.0  # obj BCELoss positive_weight
    iou_t = 0.20  # iou training threshold
    lr0 = 0.01  # initial learning rate (SGD=5E-3 Adam=5E-4)
    lrf = 0.0005  # final learning rate (with cos scheduler)
    momentum = 0.937  # SGD momentum
    weight_decay = 0.0005  # optimizer weight decay
    fl_gamma = 0.0  # focal loss gamma (efficientDet default is gamma=1.5)
    hsv_h = 0.0138  # image HSV-Hue augmentation (fraction)
    hsv_s = 0.678  # image HSV-Saturation augmentation (fraction)
    hsv_v = 0.36  # image HSV-Value augmentation (fraction)
    degrees = 1.98 * 0  # image rotation (+/- deg)
    translate = 0.05 * 0  # image translation (+/- fraction)
    scale = 0.05 * 0  # image scale (+/- gain)
    shear = 0.641 * 0  # image shear (+/- deg)

    def __init__(self, **kwargs):
        update_attrs(self, kwargs)

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class PlaneRCNNOptions:
    numEpochs: int
    batchSize: int
    LR: float
    gpu: int = 1
    task: str = "train"  # [train, test, predict]
    anchorFolder: str = "fox/planercnn/anchors"
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

    def __init__(self, epochs, batch_size, planercnn_lr = 1e-5, **kwargs):
        self.numEpochs = epochs
        self.batchSize = batch_size
        self.LR = planercnn_lr

        update_attrs(self, kwargs)

class MidasConfig:
    DATA_PATH: str
    WEIGHTS_PATH = "weights/midas.pt"
    FREEZE = True

    def __init__(self, data, **kwargs):
        self.DATA_PATH = data
        update_attrs(self, kwargs)


class YoloV3Config:
    HEAD_CONFIG_PATH = "config/yolov3-head.cfg"
    DETECTOR_CONFIG_PATH = "config/yolov3-spp-detector.cfg"
    DETECTOR_WEIGHTS_PATH = "weights/yolo-detector.pt"

    hyp: YoloHyp

    opt: YoloV3Options

    def __init__(self, **kwargs):
        update_attrs(self, kwargs)
        self.hyp = YoloHyp(**kwargs)
        self.opt = YoloV3Options(**kwargs)


class PlaneRCNNConfig(_Config):
    EXTERNAL_EXTRACTOR = True
    NUM_CLASSES = 4
    GPU_COUNT = 1
    INIT_LOG_DIR_AND_WEIGHTS = False # preventing planercnn model from making a log directory and loading weights
    MODEL_WEIGHTS_PATH = "weights/planercnn/checkpoint-partial.pth"
    REFINE_MODEL_WEIGHTS_PATH = "weights/planercnn/checkpoint-refine.pth"
    DATA_PATH: str
    options: PlaneRCNNOptions
    IMAGE_MAX_DIM: int
    IMAGE_MIN_DIM: int

    def __init__(self, data, **kwargs):
        self.options = PlaneRCNNOptions(**kwargs)
        super(PlaneRCNNConfig, self).__init__(self.options)

        self.DATA_PATH = data
        update_attrs(self, kwargs)


class Config:
    EPOCHS = 100
    BATCH_SIZE = 16
    DATA_DIR = "data/mini"
    IMG_SIZE = 640
    MIN_IMG_SIZE = 480  # used in yolo and planercnn
    MIDAS_LOSS_WEIGHT = 1e-3
    YOLO_LOSS_WEIGHT = 1
    PLANERCNN_LOSS_WEIGHT = 1
    YOLO_HEAD_LR = 1e-4
    MIDAS_LR = 1e-4
    PLANERCNN_HEAD_LR = 1e-4
    USE_YOLO = True
    USE_PLANERCNN = True

    yolo_config: YoloV3Config
    midas_config: MidasConfig
    planercnn_config: PlaneRCNNConfig

    def __init__(self, **kwargs):
        update_attrs(self, kwargs)

        kwargs["epochs"] = self.EPOCHS
        kwargs["batch_size"] = self.BATCH_SIZE
        kwargs["image_size_min"] = self.MIN_IMG_SIZE
        kwargs["img_size_max"] = self.IMG_SIZE
        kwargs["img_size_test"] = self.IMG_SIZE
        kwargs["IMAGE_MAX_DIM"] = self.IMG_SIZE
        kwargs["IMAGE_MIN_DIM"] = self.MIN_IMG_SIZE

        self.yolo_config = YoloV3Config(**{
            "data": os.path.join(self.DATA_DIR, "yolo", "custom.data"),
            **kwargs
        })

        self.planercnn_config = PlaneRCNNConfig(**{
            "data": os.path.join(self.DATA_DIR, "planercnn", "custom.data"),
            **kwargs
        })

        self.midas_config = MidasConfig(**{
            "data": os.path.join(self.DATA_DIR, "midas", "custom.data"),
            **kwargs
        })
        
