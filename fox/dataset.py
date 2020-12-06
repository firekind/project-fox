import os

import albumentations as A
import cv2
import numpy as np
import torch
from athena.utils.transforms import ToTensor, ToNumpy
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.folder import pil_loader

import fox.planercnn.utils as utils
from fox.planercnn.visualize_utils import image_to_mask
from fox.utils import parse_data_cfg
from fox.yolov3.utils.datasets import LoadImagesAndLabels as YoloDataset
from fox.yolov3.utils.parse_config import parse_data_cfg as parse_yolo_data_cfg


class PlaneRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super(PlaneRCNNDataset, self).__init__()

        self.config = config
        self.split = "train"

        data_dict = parse_data_cfg(config.DATA_PATH)
        self.mask_path = data_dict["masks"]
        self.parameter_path = data_dict["parameters"]
        self.images_path = data_dict["images"]
        self.camera_path = data_dict["camera"]

        self.samples = []
        self.img_path_to_idx = {}
        for i, f in enumerate(os.listdir(self.images_path)):
            image_path = os.path.join(self.images_path, f)
            mask_path = os.path.join(
                self.mask_path, os.path.splitext(f)[0] + "_masks.png"
            )
            param_path = os.path.join(
                self.parameter_path, os.path.splitext(f)[0] + "_parameters.npy"
            )
            self.samples.append((image_path, mask_path, param_path))
            self.img_path_to_idx[image_path] = i

        self.anchors = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            config.BACKBONE_SHAPES,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE,
        )

        with open(self.camera_path, "r") as f:
            cam_data = f.readlines()
        self.camera = np.array(
            list(map(float, cam_data[0].strip().split())), dtype=np.float32
        )

    def get_image(self, path):
        return self[self.img_path_to_idx[path]]

    def __getitem__(self, index):
        image = cv2.imread(self.samples[index][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_img = cv2.imread(self.samples[index][1])
        planes = np.load(self.samples[index][2])
        segmentation = image_to_mask(seg_img, planes.shape[0])

        instance_masks = []
        class_ids = []
        parameters = []

        normal_anchors = None
        plane_normals = None

        if len(planes) > 0:
            if self.config.ANCHOR_TYPE == "normal":
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                distances_N = np.linalg.norm(
                    np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS,
                    axis=-1,
                )
                normal_anchors = distances_N.argmin(-1)
            else:
                raise NotImplementedError(
                    f"{self.config.ANCHOR_TYPE} is not supported yet."
                )

        for plane_idx, _ in enumerate(planes):
            m = segmentation == plane_idx
            if m.max() == False:  # faster than doing m.sum() > 1
                continue

            instance_masks.append(m)

            if self.config.ANCHOR_TYPE == "normal":
                class_ids.append(normal_anchors[plane_idx] + 1)
                normal = (
                    plane_normals[plane_idx]
                    - self.config.ANCHOR_NORMALS[normal_anchors[plane_idx]]
                )
                # parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))
                parameters.append(normal)
            else:
                raise NotImplementedError(
                    f"{self.config.ANCHOR_TYPE} is not supported yet."
                )

        parameters = np.array(parameters, dtype=np.float32)
        mask = np.stack(instance_masks, axis=2)

        class_ids = np.array(class_ids, dtype=np.int32)
        depth = np.zeros(
            (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM), dtype=np.float32
        )  # dummy, not used

        (
            image,
            image_metas,
            gt_class_ids,
            gt_boxes,
            gt_masks,
            gt_parameters,
        ) = load_image_gt(
            self.config,
            0,
            image,
            depth,
            mask,
            class_ids,
            parameters,
            # augment=self.split == "train",
            augment=False
        )

        ## RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(
            image.shape, self.anchors, gt_class_ids, gt_boxes, self.config
        )

        ## If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]),
                self.config.MAX_GT_INSTANCES,
                replace=False,
            )
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
            gt_parameters = gt_parameters[ids]

        rpn_match = rpn_match[:, np.newaxis]
        image = utils.mold_image(image.astype(np.float32), self.config)

        segmentation = np.concatenate(
            [
                np.full((80, 640), fill_value=-1, dtype=np.int32),
                segmentation,
                np.full((80, 640), fill_value=-1, dtype=np.int32),
            ],
            axis=0,
        )
        extrinsics = np.eye(4, dtype=np.float32)  # dummy, hopefully wont cause issues.

        info = [
            image.transpose((2, 0, 1)).astype(np.float32),
            image_metas,
            rpn_match,
            rpn_bbox.astype(np.float32),
            gt_class_ids,
            gt_boxes.astype(np.float32),
            gt_masks.transpose((2, 0, 1)).astype(np.float32),
            gt_parameters,
            depth.astype(np.float32),
            extrinsics,
            segmentation,
            self.camera,
        ]

        return info

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        for i, sample in enumerate(batch):
            for j, data in enumerate(sample):
                if j != len(sample) - 1:  # avoiding unsqueezing camera
                    batch[i][j] = torch.from_numpy(data).unsqueeze(0)
                else:
                    batch[i][j] = torch.from_numpy(data)

        return batch


class MidasDataset(torch.utils.data.Dataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, config):
        self.config = config.midas_config

        data_dict = parse_data_cfg(self.config.DATA_PATH)
        self.depth_path = data_dict["depth"]
        self.images_path = data_dict["images"]

        self.samples = []
        self.img_path_to_idx = {}

        for i, f in enumerate(os.listdir(self.images_path)):
            image_path = os.path.join(self.images_path, f)
            depth_path = os.path.join(self.depth_path, os.path.splitext(f)[0] + ".png")
            self.samples.append((image_path, depth_path))
            self.img_path_to_idx[image_path] = i

        self.transform = A.Compose(
            [
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.Normalize(mean=MidasDataset.mean, std=MidasDataset.std, max_pixel_value=1),
                A.Lambda(ToTensor, name="ToTensor"),
            ]
        )

        self.depth_transform = A.Compose(
            [
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.Lambda(ToTensor, name="ToTensor"),
            ]
        )

    def get_image(self, path):
        return self[self.img_path_to_idx[path]]

    def __getitem__(self, index):
        img = np.array(pil_loader(self.samples[index][0]), dtype=np.float32) / 255.0
        img = self.transform(image=img)["image"]

        depth = cv2.imread(self.samples[index][1]).astype(np.float32) / 255.0
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = self.depth_transform(image=depth)["image"]

        return img, depth

    def __len__(self):
        return len(self.samples)


class ComboDataset(torch.utils.data.Dataset):
    def __init__(self, config, train=True):
        data_dict = parse_yolo_data_cfg(config.yolo_config.opt.data)
        self.yolo_dataset = YoloDataset(
            data_dict["train"] if train else data_dict["valid"],
            config.IMG_SIZE if train else config.yolo_config.opt.img_size[-1],
            config.BATCH_SIZE,
            augment=False,
            hyp=config.yolo_config.hyp,
            rect=config.yolo_config.opt.rect if train else True,
            cache_images=config.yolo_config.opt.cache_images,
            single_cls=config.yolo_config.opt.single_cls,
            mosiac=config.yolo_config.opt.mosiac,
            label_files_path=data_dict["labels"],
        )

        self.midas_dataset = MidasDataset(config)
        self.planercnn_dataset = PlaneRCNNDataset(config.planercnn_config)

        # transform - using midas mean and std values
        self.transform = A.Compose(
            [
                A.Normalize(
                    MidasDataset.mean, MidasDataset.std, max_pixel_value=1.0
                ),
            ]
        )

        self.config = config
        self.train = train

    def __getitem__(self, index):
        # getting data from yolo dataset
        yolo_data = self.yolo_dataset[index]

        # getting image path from yolo data
        path = os.path.abspath(yolo_data[-3])
        
        # getting midas data
        midas_data = self.midas_dataset.get_image(path)

        if self.config.USE_PLANERCNN:
            # getting planercnn data
            planercnn_data = self.planercnn_dataset.get_image(path)
        else:
            planercnn_data = None

        # getting input image from yolo
        img, _, _, _, _ = yolo_data

        # converting to float
        img = img.float() / 255

        # transforming
        img = self.transform(image=ToNumpy(img))["image"]

        return torch.from_numpy(img.transpose(2, 0, 1)), midas_data, yolo_data, planercnn_data

    def __len__(self):
        return len(self.yolo_dataset)

    @staticmethod
    def collate_fn(batch):
        orig_img, midas_data, yolo_data, planercnn_data = zip(*batch)  # transposed
        img, label, path, shapes, pad = zip(*yolo_data)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        
        if planercnn_data.count(None) != len(planercnn_data):
            planercnn_data = PlaneRCNNDataset.collate_fn(planercnn_data)
        if midas_data.count(None) != len(midas_data):
            midas_data = default_collate(midas_data)

        return (
            torch.stack(orig_img),
            midas_data,
            (
                torch.stack(img, 0),
                torch.cat(label, 0),
                path,
                shapes,
                pad,
            ),
            planercnn_data,
        )


def load_image_gt(
    config,
    image_id,
    image,
    depth,
    mask,
    class_ids,
    parameters,
    augment=False,
    use_mini_mask=True,
):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    ## Load image and mask
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MAX_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING,
    )

    mask = utils.resize_mask(mask, scale, padding)

    ## Random horizontal flips.
    if augment and False:
        if np.random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            depth = np.fliplr(depth)
            pass
        pass

    ## Bounding boxes. Note that some boxes might be all zeros
    ## if the corresponding mask got cropped out.
    ## bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    ## Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        pass

    active_class_ids = np.ones(config.NUM_CLASSES, dtype=np.int32)
    ## Image meta data
    image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    if config.NUM_PARAMETER_CHANNELS > 0:
        if config.OCCLUSION:
            depth = utils.resize_mask(depth, scale, padding)
            mask_visible = utils.minimize_mask(bbox, depth, config.MINI_MASK_SHAPE)
            mask = np.stack([mask, mask_visible], axis=-1)
        else:
            depth = np.expand_dims(depth, -1)
            depth = utils.resize_mask(depth, scale, padding).squeeze(-1)
            depth = utils.minimize_depth(bbox, depth, config.MINI_MASK_SHAPE)
            mask = np.stack([mask, depth], axis=-1)
            pass
        pass
    return image, image_meta, class_ids, bbox, mask, parameters


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    ## RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    ## RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    ## Handle COCO crowds
    ## A crowd box in COCO is a bounding box around several instances. Exclude
    ## them from training. A crowd box is given a negative class ID.
    no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    ## Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    ## Match anchors to GT Boxes
    ## If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    ## If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    ## Neutral anchors are those that don't match the conditions above,
    ## and they don't influence the loss function.
    ## However, don't keep any GT box unmatched (rare, but happens). Instead,
    ## match it to the closest anchor (even if its max IoU is < 0.3).
    #
    ## 1. Set negative anchors first. They get overwritten below if a GT box is
    ## matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    ## 2. Set an anchor for each GT box (regardless of IoU value).
    ## TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    ## 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    ## Subsample to balance positive and negative anchors
    ## Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ## Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ## Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        ## Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ## For positive anchors, compute shift and scale needed to transform them
    ## to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  ## index into rpn_bbox
    ## TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        ## Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        ## Convert coordinates to center plus width/height.
        ## GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        ## Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        ## Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        ## Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox
