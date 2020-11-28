from typing import List, Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

import pytorch_lightning as pl
import numpy as np

from .losses import rmse_loss

from fox.midas.midas.midas_net import MidasNet
from fox.planercnn.eva5_helper import PlaneRCNNTrainer
from fox.planercnn.models.model import MaskRCNN
from fox.planercnn.models.refinement_net import RefineModel
from fox.yolov3.eva5_helper import YoloTrainer
from fox.yolov3.models import Darknet, create_modules
from fox.yolov3.utils.parse_config import parse_data_cfg, parse_model_cfg
from fox.yolov3.utils.utils import labels_to_class_weights
from fox.yolov3.utils.torch_utils import ModelEMA


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

        self.ema = ModelEMA(self)

    def forward(self, x):
        if self.training:
            yolo_head_out = self.yolo_head(x)
            return self.yolo_detector(yolo_head_out, x)
        else:
            yolo_head_out = self.ema.ema.yolo_head(x)
            return self.ema.ema.yolo_detector(yolo_head_out, x)

    def update_attr(self):
        self.ema.update_attr(self)

    def update_ema(self):
        self.ema.update(self)


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

            self.yolo_part = YoloPart(
                yolo_config.HEAD_CONFIG_PATH,
                yolo_config.DETECTOR_CONFIG_PATH,
                yolo_config.DETECTOR_WEIGHTS_PATH
            )

            # creating detector
            self.yolo_part.yolo_detector = Darknet(yolo_config.DETECTOR_CONFIG_PATH)
            self.yolo_part.yolo_detector.gr = yolo_config.opt.gr
            self.yolo_part.yolo_detector.hyp = yolo_config.hyp
            self.yolo_part.yolo_detector.nc = nc
            self.yolo_part.yolo_detector.class_weights = labels_to_class_weights(yolo_labels, nc)

            # creating trainer
            self.yolo_trainer = YoloTrainer(
                self.yolo_detector,
                yolo_config.hyp,
                yolo_config.opt,
                nb,  # number of batches
                nc,
            )

    def forward(self, x):
        # forward proping midas
        l1, l2, l3, l4, midas_out = self.midas_net(x)

        # forward proping yolo
        if self.config.USE_YOLO:
            yolo_head_out = self.yolo_head(l2)
            yolo_out = self.yolo_detector(yolo_head_out, l2)
        else:
            yolo_out = None

        return midas_out, yolo_out, None

    def training_step(self, batch, batch_idx):
        # getting data
        imgs, midas_data, yolo_data, planercnn_data = batch

        # forward prop
        midas_out, yolo_out, planercnn_out = self(imgs)

        # after forward prop (yolo loss)
        yolo_loss, yolo_loss_items = (
            self.yolo_trainer.post_train_step(
                yolo_out, yolo_data, batch_idx, self.current_epoch
            )
            if self.config.USE_YOLO
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
        )

        # logging
        self.log("total loss", loss.detach(), prog_bar=True)
        self.log("midas loss", midas_loss.detach(), prog_bar=True)
        if self.config.USE_YOLO:
            self.log("yolo loss", yolo_loss.detach(), prog_bar=True)

        # metrics to return
        metrics = {
            "loss": loss,
            "midas_loss": midas_loss.detach().cpu().numpy(),
        }

        if self.config.USE_YOLO:
            metrics.update(
                {"yolo_loss": yolo_loss.detach().cpu().numpy(),}
            )

        return metrics

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.config.USE_YOLO:
            self.yolo_part.update_attr()
            avg_yolo_loss = np.mean([d["yolo_loss"] for d in outputs])
            self.log("avg yolo loss", avg_yolo_loss, prog_bar=True)

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

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int, optimizer_closure: Optional[Callable], on_tpu: bool, using_native_amp: bool, using_lbfgs: bool) -> None:
        if self.config.USE_YOLO:
            if self.yolo_trainer.calc_ni(batch_idx, epoch) % self.yolo_trainer.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                self.yolo_part.update()

    def configure_optimizers(self):
        param_groups = []

        if self.config.USE_YOLO:
            # yolo param groups
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            for k, v in dict(self.yolo_part.named_parameters()).items():
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

        # midas param groups
        # param_groups.append(
        #     {
        #         "params": self.midas_net.parameters(),
        #         "lr": self.config.MIDAS_LR,
        #         "momentum": 0.9,
        #     }
        # )

        # creating optimizer
        optimizer = optim.SGD(param_groups)

        if self.config.USE_YOLO:
            # setting the optimizer for the yolo trainer
            self.yolo_trainer.set_optimizer(optimizer)

        return optimizer
