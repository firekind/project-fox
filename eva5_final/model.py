from typing import List, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from .midas.midas.midas_net import MidasNet
from .yolov3.models import create_modules, Darknet
from .yolov3.utils.parse_config import parse_model_cfg, parse_data_cfg
from .yolov3.utils.utils import labels_to_class_weights
from .yolov3.eva5_helper import YoloTrainer


class YoloHead(torch.nn.Module):
    def __init__(self, config):
        super(YoloHead, self).__init__()
        # parsing config file
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


class Model(pl.LightningModule):
    def __init__(self, config, yolo_labels, nb):
        super(Model, self).__init__()

        self.config = config
        midas_config = config.midas_config
        yolo_config = config.yolo_config

        #############################
        # MIDAS
        #############################

        # creating midas model
        self.midas_net = MidasNet(midas_config.WEIGHTS_PATH)

        # freezing midas model
        if midas_config.FREEZE:
            for param in self.midas_net.parameters():
                param.requires_grad = False

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
        self.yolo_detector.gr = yolo_config.gr
        self.yolo_detector.hyp = yolo_config.hyp
        self.yolo_detector.nc = nc
        self.yolo_detector.class_weights = labels_to_class_weights(yolo_labels, nc)

        # loading detector weights
        if yolo_config.DETECTOR_WEIGHTS_PATH is not None:
            self.yolo_detector.load_state_dict(
                torch.load(yolo_config.DETECTOR_WEIGHTS_PATH)["model"], strict=False
            )

        # creating optimizer for yolo
        self.yolo_optim = self.configure_yolo_optimizer(
            yolo_config.hyp, yolo_config.opt
        )

        # creating trainer
        self.yolo_trainer = YoloTrainer(
            self.yolo_detector,
            self.yolo_optim,
            yolo_config.hyp,
            yolo_config.opt,
            nb,  # number of batches
            nc,
        )

    def forward(self, x):
        # forward proping midas
        l1, l2, l3, l4, midas_out = self.midas_net(x)

        # forward proping yolo
        yolo_head_out = self.yolo_head(l2)
        yolo_out = self.yolo_detector(yolo_head_out, l2)

        # TODO: forward proping planercnn
        
        return midas_out, yolo_out, None

    def configure_optimizers(self):
        lf = (
            lambda x: (((1 + np.cos(x * np.pi / self.epochs)) / 2) ** 1.0) * 0.95 + 0.05
        )  # cosine
        yolo_scheduler = LambdaLR(self.yolo_optim, lr_lambda=lf)

        return [self.yolo_optim,], [yolo_scheduler,]

    def training_step(self, batch, batch_idx):
        img, midas_data, yolo_data, planercnn_data = batch

        self.yolo_trainer.pre_train_step(
            yolo_data, batch_idx, self.current_epoch
        )  # ignoring multi-scale changes (returned by pre_train_step)

        # forward prop
        midas_out, yolo_out, planercnn_out = self(img)

        yolo_loss = self.yolo_trainer.post_train_step(
            yolo_out, yolo_data, batch_idx, self.current_epoch
        )

        self.log("yolo loss", yolo_loss, prog_bar=True)

        # return loss
        return {
            # "loss": loss,
            "yolo_loss": yolo_loss
        }

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.yolo_trainer.train_epoch_end()
        avg_yolo_loss = np.mean([d["yolo_loss"] for d in outputs])

        self.log("avg yolo loss", avg_yolo_loss, prog_bar=True)

        # log stuff

    def on_validation_epoch_start(self) -> None:
        self.yolo_trainer.validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        img, midas_data, yolo_data, planercnn_data = batch
        midas_out, yolo_out, planercnn_out = self(img)

        #############################
        # YoloV3
        #############################
        yolo_loss = self.yolo_trainer.validation_step(
            self.config.yolo_config.opt,
            yolo_out,
            yolo_data,
            batch_idx,
            self.current_epoch,
        )
        self.log("yolo_val_loss", yolo_loss)
        
        return {
            # loss: loss,
            'yolo_val_loss': yolo_loss
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        #############################
        # YoloV3
        #############################

        # `mAPs` is avg. precision for each class, `mAP` is total mAP
        (mp, mr, mAP, mf1), mAPs = self.yolo_trainer.validation_epoch_end()
        avg_yolo_val_loss = np.mean([d['yolo_val_loss'] for d in outputs])

        # log stuff
        self.log("avg yolo val loss", avg_yolo_val_loss, prog_bar=True)
        self.log("yolo mAP", mAP, prog_bar=True)
        self.log("yolo mean precision", mp)
        self.log("yolo mean recall", mr)
        self.log("yolo mean f1", mf1)

    def configure_yolo_optimizer(self, hyp, opt):
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.yolo_detector.named_parameters()).items():
            if ".bias" in k:
                pg2 += [v]  # biases
            elif "Conv2d.weight" in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        if opt.adam:
            # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
            optimizer = optim.Adam(pg0, lr=hyp["lr0"])
        else:
            optimizer = optim.SGD(
                pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
            )

        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})
        del pg0, pg1, pg2

        return optimizer

