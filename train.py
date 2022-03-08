import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from packaging import version

from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d, CrossEntropy2d_pnd
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet, VOCDataSet_CLS, VOCGTDataSet_npy
from utils.eps import get_eps_loss


from utils.metric import scores
from tqdm import tqdm

import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 10
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'  #your path for VOC2012
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 12000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './model/resnet101COCO-41f33a49.pth'
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4

LAMBDA_ADV= 0.001


high_T = 1.2
low_T = 0.8


def get_arguments():
    
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--high-T", type=float, default=high_T,
                        help="high T.")
    parser.add_argument("--low-T", type=float, default=low_T,
                        help="low T.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def loss_calc_pnd(pred, label, gpu, thr_T=0.8):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d_pnd().cuda(gpu)

    return criterion(pred, label, thr_T)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label


def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    bestIoU = 0

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)

    # load pretrained parameters
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print(name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copy {}'.format(name))
    model.load_state_dict(new_params)


    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cuda(args.gpu)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_dataset_cls = VOCDataSet_CLS(args.data_dir, './dataset/voc_list/train_cls.txt', crop_size=input_size,
                                       mirror=args.random_mirror, mean=IMG_MEAN, use_cache=False)

    train_dataset = VOCDataSet(args.data_dir, './dataset/voc_list/simple_list.txt', crop_size=input_size,
                               scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, use_cache=False)
    train_complex_dataset = VOCDataSet(args.data_dir, './dataset/voc_list/complex_list.txt', crop_size=input_size,
                                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                       use_cache=False)

    train_gt_dataset = VOCGTDataSet(args.data_dir, './dataset/voc_list/simple_list.txt', crop_size=input_size,
                                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, use_cache=False)

    
    trainloader_cls = data.DataLoader(train_dataset_cls,
                    batch_size=args.batch_size,num_workers=3, pin_memory=True,shuffle=True,)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=args.batch_size,num_workers=3, pin_memory=True,shuffle=True,)
    trainloader_complex = data.DataLoader(train_complex_dataset,
                    batch_size=args.batch_size, num_workers=3, pin_memory=True,shuffle=True,)
    trainloader_gt = data.DataLoader(train_gt_dataset,
                    batch_size=args.batch_size, num_workers=3, pin_memory=True,shuffle=True,)

    trainloader_complex_iter = enumerate(trainloader_complex)


    trainloader_cls_iter = enumerate(trainloader_cls)

    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)


    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')


    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    ada_iou_thr = [0.4,0.5,0.5,0.5,0.5,
                   0.6,0.6,0.6,0.4,0.6,
                   0.0,0.7,0.6,0.5,0.6,
                   0.4,0.6,0.4,0.5,0.4]    # We directly provide the class adaptive thresholds for simplicity. Floor function is performed for stable reproducibility.


    for i_iter in range(args.num_steps):
        loss_cls_value = 0
        loss_seg_value = 0
        loss_D_value = 0
        loss_adv_simple_value = 0
        loss_adv_complex_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # adversarial loss for complex sets
            try:
                _, batch = next(trainloader_complex_iter)
            except:
                trainloader_complex_iter = enumerate(trainloader_complex)
                _, batch = next(trainloader_complex_iter)

            images, labels, class_ids, _, _ = batch
            images = Variable(images).cuda(args.gpu)
            pred = interp(model(images))
            pred_complex = pred.detach()
            D_out = interp(model_D(F.softmax(pred)))
            D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)
            ignore_mask_complex = np.zeros(D_out_sigmoid.shape).astype(np.bool)
            loss_adv_complex = args.lambda_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_complex))
            loss_adv_complex = loss_adv_complex/args.iter_size
            loss_adv_complex_value += loss_adv_complex.data.cpu().numpy()/args.lambda_adv
            loss_adv_complex.backward()


            #classification loss
            try:
                _, batch = next(trainloader_cls_iter)
            except:
                trainloader_cls_iter = enumerate(trainloader_cls)
                _, batch = next(trainloader_cls_iter)
            
            images, labels = batch
            images = Variable(images).cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            logits = model(images)                    
            logits = torch.mean(logits, dim=(2,3))
            logits = logits[:,1:]

            if len(logits.shape) == 1:
                logits = logits.reshape(labels.shape)
            loss_cls = F.multilabel_soft_margin_loss(logits, labels)

            loss_cls = loss_cls/args.iter_size
            loss_cls.backward()
            loss_cls_value += loss_cls.data.cpu().numpy()/args.iter_size
            
            
            # train with simple images

            try:
                _, batch = next(trainloader_iter)
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = next(trainloader_iter)

            images, labels, class_ids, _, _ = batch
            class_ids = class_ids.long().cuda(args.gpu)
            images = Variable(images).cuda(args.gpu)
            ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images))                 

            #class adaptive thresholds 
            if i_iter >=1000:
                with torch.no_grad():
                    b, _, _, _ = pred.size()
                    role_1 = pred.argmax(axis=1)
                    role_1 = (role_1!=0).detach()
                    role_2 =  ((labels>0)&(labels<255)).detach().cuda(args.gpu)

                    iou_role = (role_1&role_2).type(torch.float32).view(b, -1).sum(-1) / \
                        ((role_1|role_2).type(torch.float32) + 1e-04).view(b, -1).sum(-1)


                thres = torch.zeros(b, dtype=torch.float).cuda(args.gpu)
                for i in range(b):
                    first = class_ids[i]
                    thres[i] = ada_iou_thr[first]

                if (iou_role>thres).type(torch.float32).sum()<1:
                    thres=thres/2
                    if (iou_role>thres).type(torch.float32).sum()<1:
                        thres = -1 * torch.ones(b, dtype=torch.float).cuda(args.gpu)


                pred = pred[iou_role>thres]
                labels = labels[iou_role>thres]
                images = images[iou_role>thres]
                ignore_mask = ignore_mask[(iou_role>thres).cpu().numpy()]


            #s2c alignment with cutmix begin
            r = np.random.rand(1)
            if r < 0.5:
                lam = np.random.beta(1, 1)
                rand_index = torch.randperm(images.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_index, bbx1:bbx2, bby1:bby2]
                pred_mix = interp(model(images))

                if i_iter < 1000:
                    loss_seg = loss_calc(pred_mix, labels, args.gpu)
                else:
                    if i_iter > 11000:
                        thr_T = args.low_T
                    else:
                        thr_T = args.high_T - (i_iter/1000-1) * (args.high_T-args.low_T)/10
                    loss_seg = loss_calc_pnd(pred_mix, labels, args.gpu, thr_T)

                loss_adv_pred = 0 
            else:
                if i_iter < 1000:
                    loss_seg = loss_calc(pred, labels, args.gpu)
                else:
                    if i_iter > 11000:
                        thr_T = args.low_T
                    else:
                        thr_T = args.high_T - (i_iter/1000-1) * (args.high_T-args.low_T)/10
                    loss_seg = loss_calc_pnd(pred, labels, args.gpu, thr_T)
               
                # adversarial loss for simple sets
                D_out = interp(model_D(F.softmax(pred)))
                loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))
                loss_adv_simple_value += loss_adv_pred.data.cpu().numpy()/args.iter_size

            
            loss = loss_seg + args.lambda_adv * loss_adv_pred

            
            loss = loss/args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy()/args.iter_size
            
            # train D
            for param in model_D.parameters():
                param.requires_grad = True

            # train with prediction
            pred = pred.detach()
            pred = torch.cat((pred, pred_complex), 0)
            ignore_mask = np.concatenate((ignore_mask,ignore_mask_complex), axis = 0)   

            D_out = interp(model_D(F.softmax(pred)))
            loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()


            # train with simple gt
            try:
                _, batch = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = next(trainloader_gt_iter)

           
            images, labels_gt, class_ids, _, _ = batch
            class_ids = class_ids.long().cuda(args.gpu)

            #class adaptive thresholds 
            if i_iter >=1000:
                with torch.no_grad():
                    images = Variable(images).cuda(args.gpu)
                    pred = interp(model(images))                 
                    b, _, _, _ = pred.size()
                    role_1 = pred.argmax(axis=1)
                    role_1 = (role_1!=0).detach()
                    role_2 =  ((labels_gt>0)&(labels_gt<255)).detach().cuda(args.gpu)

                    iou_role = (role_1&role_2).type(torch.float32).view(b, -1).sum(-1) / \
                        ((role_1|role_2).type(torch.float32) + 1e-04).view(b, -1).sum(-1)


                thres = torch.zeros(b, dtype=torch.float).cuda(args.gpu)
                for i in range(b):
                    thres[i] = ada_iou_thr[class_ids[i]]
                
                if (iou_role>thres).type(torch.float32).sum()<1:
                    thres=thres/2
                    if (iou_role>thres).type(torch.float32).sum()<1:
                        thres = -1 * torch.ones(b, dtype=torch.float).cuda(args.gpu)
                labels_gt = labels_gt[iou_role>thres]
                
            
            D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
            ignore_mask_gt = (labels_gt.numpy() == 255)

            D_out = interp(model_D(D_gt_v))
            loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()



        optimizer.step()
        optimizer_D.step()

        #print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_cls = {3:.3f}, loss_adv_simple = {4:.3f}, loss_adv_complex = {5:.3f}, loss_D = {6:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_cls_value, loss_adv_simple_value, loss_adv_complex_value, loss_D_value))

        

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            testloader = data.DataLoader(VOCGTDataSet_npy(args.data_dir, './dataset/voc_list/val.txt', mean=IMG_MEAN, scale=False, mirror=False),
                            batch_size=1, shuffle=False, pin_memory=True)
            model.eval()
            preds, gts = [], []
            for image_ids, images, gt_labels in tqdm(
                testloader, total=len(testloader), dynamic_ncols=True
            ):
                # Image
                images = images.cuda(args.gpu)

            # Forward propagation
                with torch.no_grad():
                    logits = model(images)
                    _, _, H, W = logits.shape
                    interp_val = lambda l: F.interpolate(
                        l, size=(H, W), mode="bilinear", align_corners=False
                    )

                    # Scaled
                    logits_pyramid = []
                    for p in [0.75]:
                        h = F.interpolate(images, scale_factor=p, mode="bilinear", align_corners=False)
                        logits_pyramid.append(model(h))

                    # Pixel-wise max
                    logits_all = [logits] + [interp_val(l) for l in logits_pyramid]
                    logits = torch.max(torch.stack(logits_all), dim=0)[0]

                # Pixel-wise labeling
                _, H, W = gt_labels.shape
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=True
                )
                probs = F.softmax(logits, dim=1)
                labels_pred = torch.argmax(probs, dim=1)

                preds += list(labels_pred.cpu().numpy())
                gts += list(gt_labels.numpy())

            # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
            score = scores(gts, preds, n_class=21)

            mIoU = score["Mean IoU"]

            if mIoU > bestIoU:
                bestIoU = mIoU
                print('taking snapshot ...')
                torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_best.pth'))
                torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_best_D.pth'))
            print('===> best  mIoU: ' + str(bestIoU) )
            model.train()


    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
