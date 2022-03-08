import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
from .imutils import ResizeShort
from .transforms import transforms
from tqdm import tqdm

class VOCDataSet_CLS(data.Dataset):
    def __init__(self, root_dir, datalist_file, num_classes=20, crop_size=(321, 321), mean=(128, 128, 128), mirror=True, use_cache=True):

        self.transform = transforms.Compose([transforms.Resize(321),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])
        self.root_dir = root_dir
        self.datalist_file =  datalist_file
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.mean = mean
        self.use_cache = use_cache
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.loaded_samples = self._cache_dataset() if self.use_cache else None
    
    def _cache_dataset(self):
        cached_samples = []
        print('caching samples ... ')
        for idx, path in enumerate(tqdm(self.image_list)):
            image = Image.open(path).convert('RGB')
            cached_samples.append(image)
        print(len(cached_samples))
        return cached_samples

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.use_cache:
            image = self.loaded_samples[idx]
        else:
            img_name =  self.image_list[idx]
            image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_LINEAR)
        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        
        return image.copy(),  self.label_list[idx]
    
    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = 'JPEGImages/' + fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels

    

    


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, use_cache=True):
        self.root = root
        self.use_cache = use_cache
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split()[0] for i_id in open(list_path)]
        self.class_id = [i_id.strip().split()[1] for i_id in open(list_path)]  
        
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join("./dataset/sal2gt/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.loaded_images, self.loaded_labels = self._cache_dataset() if self.use_cache else None,None
        
    
    def _cache_dataset(self):
        cached_images = []
        cached_labels = []
        print('caching samples ... ')
        for idx, datafiles in enumerate(tqdm(self.files)):
            image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
            label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
            cached_images.append(image)
            cached_labels.append(label)
        print(len(cached_images))
        print(len(cached_labels))
        return cached_images, cached_labels

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        class_id = int(self.class_id[index])

        if self.use_cache:
            image = self.loaded_images[index]
            label = self.loaded_labels[index]
        else:
            image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
            label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), class_id, np.array(size), name



class VOCGTDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, use_cache=True):
        self.root = root
        self.use_cache = use_cache
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        #self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = [i_id.strip().split()[0] for i_id in open(list_path)]
        self.class_id = [i_id.strip().split()[1] for i_id in open(list_path)]  #only for simplex image that contains one category

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join("./dataset/sal2gt/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.loaded_images, self.loaded_labels = self._cache_dataset() if self.use_cache else None,None
    
    def _cache_dataset(self):
        cached_images = []
        cached_labels = []
        print('caching samples ... ')
        for idx, datafiles in enumerate(tqdm(self.files)):
            image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
            label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
            cached_images.append(image)
            cached_labels.append(label)
        print(len(cached_images))
        print(len(cached_labels))
        return cached_images, cached_labels

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        class_id = int(self.class_id[index])
        if self.use_cache:
            image = self.loaded_images[index]
            label = self.loaded_labels[index]
        else:
            image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
            label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        attempt = 0
        while attempt < 10 :
            if self.scale:
                image, label = self.generate_scale_label(image, label)

            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                attempt += 1
                continue
            else:
                break

        if attempt == 10 :
            image = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, self.crop_size, interpolation = cv2.INTER_NEAREST)


        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), class_id, np.array(size), name



class VOCGTDataSet_crf(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(datafiles["label"]), dtype=np.int32)
        name = datafiles["name"]

        return name, image, label

class VOCGTDataSet_npy(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = np.asarray(Image.open(datafiles["label"]), dtype=np.int32)
        name = datafiles["name"]


        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        return name, image.copy(), label




if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
