import enum
from typing import Any, List, Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .losses import rmse_loss
from .midas.midas.midas_net import MidasNet
from .planercnn.eva5_helper import PlaneRCNNTrainer
from .planercnn.models.model import MaskRCNN
from .planercnn.models.refinement_net import RefineModel
from .yolov3.eva5_helper import YoloTrainer
from .yolov3.models import Darknet, create_modules
from .yolov3.utils.parse_config import parse_data_cfg, parse_model_cfg
from .yolov3.utils.utils import labels_to_class_weights


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


class Model(pl.LightningModule):
    def __init__(self, config, yolo_labels, nb):
        super(Model, self).__init__()

        assert (
            config.USE_YOLO != False or config.USE_PLANERCNN != False
        ), "Use one of yolo or planercnn, or both."

        self.config = config
        midas_config = config.midas_config
        yolo_config = config.yolo_config
        planercnn_config = config.planercnn_config

        #############################
        # MIDAS
        #############################

        # creating midas model
        self.midas_net = MidasNet(midas_config.WEIGHTS_PATH)

        # freezing midas model
        if midas_config.FREEZE:
            for param in self.midas_net.parameters():
                param.requires_grad = False

        if config.USE_YOLO:
            #############################
            # YoloV3
            #############################

            # getting number of classes
            data_dict = parse_data_cfg(yolo_config.opt.data)
            nc = 1 if yolo_config.opt.single_cls else int(data_dict["classes"])

            # creating the head
            self.yolo_head = YoloHead(yolo_config.HEAD_CONFIG_PATH)

            # creating detector
            self.yolo_detector = Darknet(yolo_config.DETECTOR_CONFIG_PATH)
            self.yolo_detector.gr = yolo_config.opt.gr
            self.yolo_detector.hyp = yolo_config.hyp
            self.yolo_detector.nc = nc
            self.yolo_detector.class_weights = labels_to_class_weights(yolo_labels, nc)

            # loading detector weights
            if yolo_config.DETECTOR_WEIGHTS_PATH is not None:
                self.yolo_detector.load_state_dict(
                    torch.load(yolo_config.DETECTOR_WEIGHTS_PATH)["model"], strict=False
                )

            # creating trainer
            self.yolo_trainer = YoloTrainer(
                self.yolo_detector,
                yolo_config.hyp,
                yolo_config.opt,
                nb,  # number of batches
                nc,
            )

        if config.USE_PLANERCNN:
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

            # creating trainer
            self.planercnn_trainer = PlaneRCNNTrainer(
                planercnn_config, self.planercnn_refine_model
            )

    def forward(self, x, planercnn_data=None):
        # forward proping midas
        l1, l2, l3, l4, midas_out = self.midas_net(x)

        # forward proping yolo
        if self.config.USE_YOLO:
            yolo_head_out = self.yolo_head(l2)
            yolo_out = self.yolo_detector(yolo_head_out, l2)
        else:
            yolo_out = None

        # forward proping planercnn's maskrcnn
        if planercnn_data is not None and self.config.USE_PLANERCNN:
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

    def training_step(self, batch, batch_idx):
        # getting data
        imgs, midas_data, yolo_data, planercnn_data = batch

        # transfering planercnn data to device
        for i, sample in enumerate(planercnn_data):
            for j, data in enumerate(sample):
                planercnn_data[i][j] = data.to(self.device)

        (
            _,  # images,
            image_metas,
            _,  # rpn_match,
            _,  # rpn_bbox,
            gt_class_ids,
            gt_boxes,
            gt_masks,
            gt_parameters,
            _,  # gt_depth,
            _,  # extrinsics,
            _,  # gt_segmentation,
            camera,
        ) = zip(*planercnn_data)

        # forward prop
        midas_out, yolo_out, planercnn_out = self(
            imgs,
            zip(image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera)
            if self.config.USE_PLANERCNN
            else None,
        )

        # after forward prop (yolo loss)
        yolo_loss, yolo_loss_items = (
            self.yolo_trainer.post_train_step(
                yolo_out, yolo_data, batch_idx, self.current_epoch
            )
            if self.config.USE_YOLO
            else 0
        )

        # planercnn loss
        planercnn_loss = (
            torch.mean(
                torch.stack(
                    self.planercnn_trainer.train_step_on_batch(
                        planercnn_data, planercnn_out, device=self.device
                    ),
                    0,
                )
            )
            if self.config.USE_PLANERCNN
            else 0
        )

        # midas loss
        midas_loss = rmse_loss(
            midas_out, midas_data[-1], letterbox_borders=yolo_data[-1]
        )

        # total loss
        loss = (
            self.config.MIDAS_LOSS_WEIGHT * midas_loss
            + self.config.YOLO_LOSS_WEIGHT * yolo_loss
            + self.config.PLANERCNN_LOSS_WEIGHT * planercnn_loss
        )

        self.manual_backward(loss, self.optimizers())

        if self.config.USE_YOLO:
            if self.yolo_trainer.calc_ni(batch_idx, self.current_epoch) % self.yolo_trainer.accumulate:
                    self.optimizers().step()
                    self.optimizers().zero_grad()
                    self.yolo_trainer.update_ema()

        # logging
        self.log("total loss", loss, prog_bar=True)
        self.log("midas loss", midas_loss, prog_bar=True)
        if self.config.USE_YOLO:
            self.log("yolo loss", yolo_loss, prog_bar=True)
        if self.config.USE_PLANERCNN:
            self.log("planercnn loss", planercnn_loss, prog_bar=True)

        # metrics to return
        metrics = {
            "loss": loss,
            "midas_loss": midas_loss.detach().cpu().numpy(),
        }

        if self.config.USE_YOLO:
            metrics.update(
                {"yolo_loss": yolo_loss.detach().cpu().numpy(),}
            )
        if self.config.USE_PLANERCNN:
            metrics.update(
                {"planercnn_loss": planercnn_loss.detach().cpu().numpy(),}
            )

        return metrics

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.config.USE_YOLO:
            self.yolo_trainer.train_epoch_end()
            avg_yolo_loss = np.mean([d["yolo_loss"] for d in outputs])
            self.log("avg yolo loss", avg_yolo_loss, prog_bar=True)

        if self.config.USE_PLANERCNN:
            avg_planercnn_loss = np.mean([d["planercnn_loss"] for d in outputs])
            self.log("avg planercnn loss", avg_planercnn_loss, prog_bar=True)

        avg_midas_loss = np.mean([d["midas_loss"] for d in outputs])
        self.log("avg midas loss", avg_midas_loss, prog_bar=True)

        avg_total_loss = np.mean([d["loss"].detach().cpu().numpy() for d in outputs])
        self.log("avg total loss", avg_total_loss, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        if self.config.USE_YOLO:
            self.yolo_trainer.validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        imgs, _, yolo_data, _ = batch
        _, yolo_out, _ = self(imgs)

        metrics = {}

        #############################
        # YoloV3
        #############################

        if self.config.USE_YOLO:
            yolo_losses = self.yolo_trainer.validation_step(
                self.config.yolo_config.opt,
                yolo_out,
                yolo_data,
                batch_idx,
                self.current_epoch,
            )
            self.log("yolo_val_loss", yolo_losses.sum())
            metrics.update({"yolo_val_losses": yolo_losses.cpu().numpy()})

        return metrics

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        #############################
        # YoloV3
        #############################

        if self.config.USE_YOLO:
            # `mAPs` is avg. precision for each class, `mAP` is total mAP
            (mp, mr, mAP, mf1), mAPs = self.yolo_trainer.validation_epoch_end()
            avg_yolo_val_loss = np.mean([d["yolo_val_losses"].sum() for d in outputs])
            print(mAP)

            # log stuff
            self.log("avg yolo val loss", avg_yolo_val_loss, prog_bar=True)
            self.log("yolo mAP", mAP, prog_bar=True)
            self.log("yolo mean precision", mp)
            self.log("yolo mean recall", mr)
            self.log("yolo mean f1", mf1)

    def configure_optimizers(self):
        param_groups = []

        if self.config.USE_YOLO:
            # yolo param groups
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            for k, v in dict(self.yolo_detector.named_parameters()).items():
                if ".bias" in k:
                    pg2 += [v]  # biases
                elif "Conv2d.weight" in k:
                    pg1 += [v]  # apply weight_decay
                else:
                    pg0 += [v]  # all else

            hyp = self.config.yolo_config.hyp
            param_groups.append(
                {
                    "params": pg0,
                    "lr": hyp["lr0"],
                    "momentum": hyp["momentum"],
                    "nesterov": True,
                }
            )
            param_groups.append(
                {
                    "params": pg1,
                    "lr": hyp["lr0"],
                    "momentum": hyp["momentum"],
                    "nesterov": True,
                    "weight_decay": hyp["weight_decay"],
                }
            )
            param_groups.append(
                {
                    "params": pg2,
                    "lr": hyp["lr0"],
                    "momentum": hyp["momentum"],
                    "nesterov": True,
                }
            )
            param_groups.append(
                {
                    "params": self.yolo_head.parameters(),
                    "lr": self.config.YOLO_HEAD_LR,
                    "momentum": 0.9,
                }
            )

        if self.config.USE_PLANERCNN:
            # planercnn param groups
            options = self.config.planercnn_config.options
            trainables_wo_bn = [
                param
                for name, param in self.planercnn_model.named_parameters()
                if param.requires_grad and not "bn" in name
            ]
            trainables_only_bn = [
                param
                for name, param in self.planercnn_model.named_parameters()
                if param.requires_grad and "bn" in name
            ]

            param_groups.append(
                {
                    "params": trainables_wo_bn,
                    "weight_decay": 0.0001,
                    "lr": options.LR,
                    "momentum": 0.9,
                }
            )
            param_groups.append(
                {"params": trainables_only_bn, "lr": options.LR, "momentum": 9}
            )
            param_groups.append(
                {
                    "params": self.planercnn_refine_model.parameters(),
                    "lr": options.LR,
                    "momentum": 0.9,
                }
            )
            param_groups.append(
                {
                    "params": self.planercnn_head.parameters(),
                    "lr": self.config.PLANERCNN_HEAD_LR,
                    "momentum": 0.9,
                }
            )

        # midas param groups
        param_groups.append(
            {
                "params": self.midas_net.parameters(),
                "lr": self.config.MIDAS_LR,
                "momentum": 0.9,
            }
        )

        # creating optimizer
        optimizer = optim.SGD(param_groups)

        if self.config.USE_YOLO:
            # setting the optimizer for the yolo trainer
            self.yolo_trainer.set_optimizer(optimizer)

        return optimizer
