import torch
import torch.nn as nn

from fox.midas.midas.midas_net import MidasNet
from fox.planercnn.eva5_helper import PlaneRCNNTrainer
from fox.planercnn.models.model import MaskRCNN
from fox.planercnn.models.refinement_net import RefineModel
from fox.yolov3.eva5_helper import YoloTrainer
from fox.yolov3.models import Darknet, create_modules
from fox.yolov3.utils.parse_config import parse_data_cfg, parse_model_cfg
from fox.yolov3.utils.utils import labels_to_class_weights


class YoloHead(nn.Module):
    def __init__(self, config):
        super(YoloHead, self).__init__()
        # parsing config (yolo cfg) file
        module_defs = parse_model_cfg(config)

        # creating modules from config file
        self.module_list, self.routs = create_modules(
            module_defs, (416, 416), config
        )  # image size (416, 416) not used, only used if exporting to onnx

    def forward(self, x):
        out = []
        orig_inp = x
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__

            if name == "WeightedFeatureFusion":
                x = module(x, out, orig_inp)
            elif name == "FeatureConcat":
                x = module(x, out)
            else:
                x = module(x)

            out.append(x if self.routs[i] else [])

        return x


class YoloPart(nn.Module):
    def __init__(self, yolo_head_config_path, yolo_detector_config_path, yolo_detector_weights_path):
        super(YoloPart, self).__init__()

        # creating the head
        self.yolo_head = YoloHead(yolo_head_config_path)

        # creating detector
        self.yolo_detector = Darknet(yolo_detector_config_path)

        # loading detector weights
        if yolo_detector_weights_path is not None:
            self.yolo_detector.load_state_dict(
                torch.load(yolo_detector_weights_path)["model"], strict=False
            )

    def forward(self, x):
        yolo_head_out = self.yolo_head(x)
        return self.yolo_detector(yolo_head_out, x)        


class PlaneRCNNHeadSegment(nn.Module):
    def __init__(self, in_channels, factor):
        super(PlaneRCNNHeadSegment, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / factor), kernel_size=1),
            nn.BatchNorm2d(int(in_channels / factor)),
            nn.ReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(int(in_channels / factor), in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class PlaneRCNNHead(nn.Module):
    def __init__(self):
        super(PlaneRCNNHead, self).__init__()

        self.seg_1 = PlaneRCNNHeadSegment(256, 2)
        self.seg_2 = PlaneRCNNHeadSegment(512, 2)
        self.seg_3 = PlaneRCNNHeadSegment(1024, 2)
        self.seg_4 = PlaneRCNNHeadSegment(2048, 2)

    def forward(self, l1, l2, l3, l4):
        return (self.seg_1(l1), self.seg_2(l2), self.seg_3(l3), self.seg_4(l4))


class Model(nn.Module):
    def __init__(
        self,
        midas_weights_path,
        yolo_head_config_path,
        yolo_detector_config_path,
        yolo_detector_weights_path,
        use_yolo=True,
        use_planercnn=False,
        planercnn_config=None,
    ):
        super(Model, self).__init__()

        self.use_yolo = use_yolo
        self.use_planercnn = use_planercnn

        #############################
        # MIDAS
        #############################

        # creating midas model
        self.midas_net = MidasNet(midas_weights_path)

        for param in self.midas_net.parameters():
            param.requires_grad = False

        if use_yolo:
            #############################
            # YoloV3
            #############################

            self.yolo_part = YoloPart(yolo_head_config_path, yolo_detector_config_path, yolo_detector_weights_path)
            

        if use_planercnn:
            #############################
            # PlaneRCNN
            #############################

            # creating mask rcnn
            self.planercnn_model = MaskRCNN(planercnn_config)
            # loading model weights
            if planercnn_config.MODEL_WEIGHTS_PATH is not None:
                self.planercnn_model.load_state_dict(
                    torch.load(planercnn_config.MODEL_WEIGHTS_PATH), strict=False
                )

            # creating refine model
            self.planercnn_refine_model = RefineModel(planercnn_config.options)
            # loading refine model weights
            if planercnn_config.REFINE_MODEL_WEIGHTS_PATH is not None:
                self.planercnn_refine_model.load_state_dict(
                    torch.load(planercnn_config.REFINE_MODEL_WEIGHTS_PATH), strict=False
                )

            # creating planercnn head
            self.planercnn_head = PlaneRCNNHead()

    def forward(self, x, planercnn_data=None, yolo_ema=None):
        # forward proping midas
        l1, l2, l3, l4, midas_out = self.midas_net(x)

        # forward proping yolo
        if self.use_yolo:
            if self.training:
                yolo_out = self.yolo_part(l2)
            else:
                yolo_out = yolo_ema(l2)
        else:
            yolo_out = None

        # forward proping planercnn's maskrcnn
        if planercnn_data is not None and self.use_planercnn:
            planercnn_out = self.planercnn_model.predict_on_batch(
                [l1, l2, l3, l4],
                planercnn_data,
                mode="training_detection",
                use_nms=2,
                use_refinement="refinement"
                in self.config.planercnn_config.options.suffix,
            )
        else:  # during validation_step and when planercnn is disabled
            planercnn_out = None

        return midas_out, yolo_out, planercnn_out
