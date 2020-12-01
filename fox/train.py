from .losses import rmse_loss
import torch
import torch.optim as optim
from .yolov3.eva5_helper import YoloTrainer
from .yolov3.utils.torch_utils import ModelEMA
from .planercnn.eva5_helper import PlaneRCNNTrainer
from tqdm import tqdm

def train(model, config, loader, val_loader, device="cpu"):
    yolo_trainer = YoloTrainer(
        model.yolo_part.yolo_detector,
        config.yolo_config.hyp,
        config.yolo_config.opt,
        len(loader),  # number of batches
        nc=4,
    )

    planercnn_trainer = PlaneRCNNTrainer(
        config.planercnn_config, model.planercnn_refine_model
    ) if config.USE_PLANERCNN else None

    yolo_ema = ModelEMA(model.yolo_part)

    optimizer = configure_optimizer(model, config)
    yolo_trainer.set_optimizer(optimizer)

    for epoch in range(config.EPOCHS):
        
        pbar = tqdm(enumerate(loader), total=len(loader))

        for batch_idx, (imgs, midas_data, yolo_data, planercnn_data) in pbar:
            pbar.set_description(f"Epoch {epoch}")

            imgs = transfer_to_device(imgs, device)
            midas_data = transfer_to_device(midas_data, device)
            yolo_data = transfer_to_device(yolo_data, device)
            planercnn_data = transfer_to_device(planercnn_data, device)

            # transfering planercnn data to device
            # for i, sample in enumerate(planercnn_data):
            #     for j, data in enumerate(sample):
            #         planercnn_data[i][j] = data.to(model.device)

            midas_loss, yolo_loss, planercnn_loss, loss = train_step(
                model,
                (imgs, midas_data, yolo_data, planercnn_data),
                yolo_trainer,
                planercnn_trainer,
                config,
                batch_idx,
                epoch,
                device
            )

            pbar.set_postfix(loss="%.4f" % float(loss), midas_loss="%.4f" % float(midas_loss), yolo_loss="%.4f" % float(yolo_loss))

            loss.backward()

            if yolo_trainer.calc_ni(batch_idx, epoch) % yolo_trainer.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                yolo_ema.update(model.yolo_part)

            break

        yolo_ema.update_attr(model.yolo_part)
        pbar.close()

        valpbar = tqdm(enumerate(val_loader), total=len(val_loader))

        # validation
        with torch.no_grad():
            model.eval()

            yolo_trainer.validation_epoch_start()
            for batch_idx, (imgs, midas_data, yolo_data, planercnn_data) in valpbar:
                metrics = {}

                imgs = transfer_to_device(imgs, device)
                midas_data = transfer_to_device(midas_data, device)
                yolo_data = transfer_to_device(yolo_data, device)
                planercnn_data = transfer_to_device(planercnn_data, device)
                
                _, yolo_out, _ = model(imgs, yolo_ema=yolo_ema.ema)

                if config.USE_YOLO:
                    yolo_losses = yolo_trainer.validation_step(
                        config.yolo_config.opt,
                        yolo_out,
                        yolo_data,
                        batch_idx,
                        epoch
                    )
                    (mp, mr, map, mf1), maps = yolo_trainer.validation_epoch_end()

                    metrics.update(dict(
                        yolo_loss="%.4f" % float(sum(yolo_losses)),
                        map="%.4f" % map
                    ))
                
                valpbar.set_postfix(**metrics)

            model.train()
            valpbar.close()

def train_step(model, batch, yolo_trainer, planercnn_trainer, config, batch_idx, epoch, device):
    imgs, midas_data, yolo_data, planercnn_data = batch

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
    midas_out, yolo_out, planercnn_out = model(
        imgs,
        # zip(image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera)
        # if config.USE_PLANERCNN
        # else None,
    )

    # after forward prop (yolo loss)
    yolo_loss, yolo_loss_items = (
        yolo_trainer.post_train_step(
            yolo_out, yolo_data, batch_idx, epoch
        )
        if config.USE_YOLO
        else 0
    )

    # planercnn loss
    planercnn_loss = (
        torch.mean(
            torch.stack(
                planercnn_trainer.train_step_on_batch(
                    planercnn_data, planercnn_out, device=device
                ),
                0,
            )
        )
        if config.USE_PLANERCNN
        else 0
    )

    # midas loss
    midas_loss = rmse_loss(
        midas_out, midas_data[-1], letterbox_borders=yolo_data[-1]
    )

    # total loss
    loss = (
        config.MIDAS_LOSS_WEIGHT * midas_loss
        + config.YOLO_LOSS_WEIGHT * yolo_loss
        + config.PLANERCNN_LOSS_WEIGHT * planercnn_loss
    )

    return midas_loss, yolo_loss, planercnn_loss, loss



def configure_optimizer(model, config):
    param_groups = []

    if config.USE_YOLO:
        # yolo param groups
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(model.yolo_part.named_parameters()).items():
            if ".bias" in k:
                pg2 += [v]  # biases
            elif "Conv2d.weight" in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        hyp = config.yolo_config.hyp
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

    return optimizer


def transfer_to_device(data, device):
    if type(data) == torch.Tensor:
        return data.to(device)

    elif type(data) == list or type(data) == tuple:
        toreturn = []

        for sample in data:
            if type(sample) == torch.Tensor:
                toreturn.append(sample.to(device))
                continue
                
            if sample is None:
                toreturn.append(None)
                continue
            
            inner_data = []
            for datum in sample:
                if type(datum) == torch.Tensor:
                    inner_data.append(datum.to(device))
                else:
                    inner_data.append(datum)
            toreturn.append(inner_data)

        return toreturn
    
    else:
        return None