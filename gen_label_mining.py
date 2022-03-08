import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']


train_lst_path = './dataset/voc_list/train_cls.txt'

seg_path = './pseudo_labels/'


save_path = './pseudo_labels/'

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    seg_name = seg_path + name + '.png'
    

    
    if not os.path.exists(seg_name):
        print('seg_name is wrong')
        return

    gt = np.asarray(Image.open(seg_name), dtype=np.int32)
    height, width = gt.shape
    all_cat = np.unique(gt).tolist()

    if len(fields) > 2:
        flag_missing = 0
        for i in range(len(fields) - 1):
            k = i + 1
            category = int(fields[k]) + 1
            cat_exist = (gt==category)
            if cat_exist.sum() < 20: 
                flag_missing = 1
                if cat_exist.sum() > 0: 
                    gt[cat_exist] = 255
        
        if flag_missing:          #if class is missing, bg is 255
            gt[gt==0] = 255

    out = gt 
    

    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)

### Parallel Mode
pool = multiprocessing.Pool(processes=16)
pool.map(gen_gt, range(len(lines)))
pool.close()
pool.join()


