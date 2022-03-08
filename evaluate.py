import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
from packaging import version

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplab import Res_Deeplab
from dataset.voc_dataset import VOCGTDataSet_npy

from PIL import Image

import matplotlib.pyplot as plt
from compute_iou import compute_iou
from utils.metric import scores
import joblib
import json
from tqdm import tqdm

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 
RESTORE_FROM = 'VOC.pth'



def get_arguments():
    
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()



def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    model = Res_Deeplab(num_classes=args.num_classes)


    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(VOCGTDataSet_npy(args.data_dir, args.data_list, mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)

    

    logit_dir = './logit_npy'
    makedirs(logit_dir)
    save_dir = './scores'
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Score dst:", save_path)
    preds, gts = [], []
    for image_ids, images, gt_labels in tqdm(
        testloader, total=len(testloader), dynamic_ncols=True
    ):
        # Image
        images = images.cuda(gpu0)

        # Forward propagation
        with torch.no_grad():
            logits = model(images)
            _, _, H, W = logits.shape
            interp = lambda l: F.interpolate(
                l, size=(H, W), mode="bilinear", align_corners=False
            )

            # Scaled
            logits_pyramid = []
            for p in [0.75]:
                h = F.interpolate(images, scale_factor=p, mode="bilinear", align_corners=False)
                logits_pyramid.append(model(h))

            # Pixel-wise max
            logits_all = [logits] + [interp(l) for l in logits_pyramid]
            logits = torch.max(torch.stack(logits_all), dim=0)[0]

        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=True
        )
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=21)

    print('meanIOU: ' + str(score["Mean IoU"]) +'\n') 

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
