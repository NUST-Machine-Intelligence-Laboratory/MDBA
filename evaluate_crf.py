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
from dataset.voc_dataset import VOCGTDataSet_crf

from PIL import Image

import matplotlib.pyplot as plt
from compute_iou import compute_iou
from utils.crf import DenseCRF
from utils.metric import scores
import joblib
import json

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
                        help="available options : DeepLab/DRN")
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


    testloader = VOCGTDataSet_crf(args.data_dir, args.data_list, mean=IMG_MEAN, scale=False, mirror=False)
                                    
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )
    

    logit_dir = './logit_npy'
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python evaluate.py")
        quit()

    # Path to save scores
    save_dir = './scores'
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)

    # Process per sample
    def process(i):
        image_id, image, gt_label = testloader.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=True)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)

        return label, gt_label

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=10, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(testloader))]
    )

    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=21)

    print('meanIOU: ' + str(score["Mean IoU"]) +'\n')

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
