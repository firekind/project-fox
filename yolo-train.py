import argparse
from copy import deepcopy

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from fox.yolov3.models import *
from fox.yolov3.utils.datasets import *
from fox.yolov3.utils.utils import *
from fox.yolov3.eva5_helper import YoloTrainer
from fox.dataset import ComboDataset
from fox.config import Config
from eva5_model import Model

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

def collate_fn(batch):
    yolo_data = LoadImagesAndLabels.collate_fn(batch)
    return yolo_data[0], None, yolo_data, None

def train(
    hyp,
    device="cuda"
):
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)
    config = Config(USE_PLANERCNN=False, BATCH_SIZE=opt.batch_size, IMG_SIZE=imgsz_max, MIN_IMG_SIZE=imgsz_min, DATA_DIR="data", cache_images=opt.cache_images)


    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    label_path = data_dict['labels']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Model(
        opt.midas_weights_path,
        opt.yolo_head_config_path,
        opt.yolo_detector_config_path,
        opt.yolo_detector_weights_path        
    ).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.yolo_part.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Dataset
    # dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
    #                               augment=True,
    #                               hyp=hyp,  # augmentation hyperparameters
    #                               rect=opt.rect,  # rectangular training
    #                               cache_images=opt.cache_images,
    #                               single_cls=opt.single_cls,
    #                               label_files_path=label_path,
    #                               mosiac=False)

    dataset = ComboDataset(config)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
                                            #  collate_fn=collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(
                                                # LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                #                  hyp=hyp,
                                                #                  rect=True,
                                                #                  cache_images=opt.cache_images,
                                                #                  single_cls=opt.single_cls,
                                                #                  label_files_path=label_path,
                                                #                  mosiac=False),
                                             ComboDataset(config, train=False),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
                                            #  collate_fn=collate_fn)

    model.yolo_part.yolo_detector.nc = nc  # attach number of classes to model
    model.yolo_part.yolo_detector.hyp = hyp  # attach hyperparameters to model
    model.yolo_part.yolo_detector.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # model.yolo_part.yolo_detector.class_weights = labels_to_class_weights(dataset.yolo_labels, nc).to(device)  # attach class weights
    model.yolo_part.yolo_detector.class_weights = labels_to_class_weights(dataset.yolo_dataset.yolo_labels, nc).to(device)  # attach class weights


    # Model EMA
    ema = torch_utils.ModelEMA(model.yolo_part)
    trainer = YoloTrainer(model.yolo_part.yolo_detector, hyp, opt, len(dataloader), nc=nc)

    # Start training
    nb = len(dataloader)  # number of batches
    for epoch in range(epochs):  # epoch ------------------------------------------------------------------
        model.train()

        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, midas_data, yolo_data, planercnn_data) in pbar:  # batch -------------------------------------------------------------
            _imgs, targets, paths, shapes, pad = yolo_data

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Forward
            midas_out, pred, planercnn_out = model(imgs)

            loss, loss_items= trainer.post_train_step(pred, (imgs, targets, paths, shapes, pad), i, epoch)
            loss.backward()

            pbar.set_postfix(loss="%.4f" % float(loss))

            # Optimize
            if trainer.calc_ni(i, epoch) % trainer.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model.yolo_part)

            # end batch ------------------------------------------------------------------------------------------------

        # Process epoch results
        ema.update_attr(model.yolo_part)
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            valpbar = tqdm(enumerate(testloader), total=len(testloader))
            with torch.no_grad():
                model.eval()

                trainer.validation_epoch_start()
                for batch_idx, (imgs, midas_data, yolo_data, planercnn_data) in valpbar:
                    _imgs, targets, paths, shapes, pad = yolo_data

                    imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                    targets = targets.to(device)
                    
                    _, yolo_out, _ = model(imgs, yolo_ema=ema.ema)

                    # if config.USE_YOLO:
                    yolo_losses = trainer.validation_step(
                        opt,
                        yolo_out,
                        (imgs, targets, paths, shapes, pad),
                        batch_idx,
                        epoch
                    )
                    

                (mp, mr, map, mf1), maps = trainer.validation_epoch_end()
                model.train()
                valpbar.close()
                print("mAP:", map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
    parser.add_argument('--midas-weights-path', type=str, default="weights/midas.pt")
    parser.add_argument('--yolo-head-config-path', type=str, default="config/yolov3-head.cfg")
    parser.add_argument('--yolo-detector-config-path', type=str, default="config/yolov3-spp-detector.cfg")
    parser.add_argument('--yolo-detector-weights-path', type=str, default="weights/yolo-detector.pt")
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.6)
    opt = parser.parse_args()
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    check_git_status()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    train(hyp)  # train normally