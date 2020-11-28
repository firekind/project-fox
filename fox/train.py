from .losses import rmse_loss
import torch
from .yolov3.eva5_helper import YoloTrainer
from .planercnn.eva5_helper import PlaneRCNNTrainer

def train(model, config, loader, val_loader):
    yolo_trainer = YoloTrainer(
        model.yolo_detector,
        config.yolo_config.hyp,
        config.yolo_config.opt,
        len(loader),  # number of batches
        nc=4,
    )

    planercnn_trainer = PlaneRCNNTrainer(
        config.planercnn_config, model.planercnn_refine_model
    ) if config.USE_PLANERCNN else None

    for epoch in config.EPOCHS:
        for batch_idx, (img, midas_data, yolo_data, planercnn_data) in enumerate(loader):
            # transfering planercnn data to device
            for i, sample in enumerate(planercnn_data):
                for j, data in enumerate(sample):
                    planercnn_data[i][j] = data.to(model.device)

            (
                imgs,  # images,
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
                torch.cat(imgs, 0),
                zip(image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera)
                if config.USE_PLANERCNN
                else None,
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
                            planercnn_data, planercnn_out, device=model.device
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