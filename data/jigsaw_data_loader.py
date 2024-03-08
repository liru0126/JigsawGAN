import os
import collections
import json
import torch
import torchvision
import glob
import numpy as np
import scipy.misc as m
import cv2
import utils
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from random import sample, random


# def get_data_path(name):
#   js = open('./config/config.json').read()
#   data = json.loads(js)
#   return data[name]['data_path']


class JigsawDataset(data.Dataset):
    def __init__(self, names, data_path=None, grid_size=3, jig_classes=10, img_transformer=None, img_tensor=None, tile_transformer=None, patches=True,
                 bias_whole_image=None):
        self.names = names
        self.data_path = data_path
        self.N = len(self.names)

        #import pdb
        #pdb.set_trace()

        self.permutations = utils.retrieve_permutations(grid_size, jig_classes)
        self.grid_size = grid_size
        self.bias_whole_image = bias_whole_image
        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._image_tensor = img_tensor
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = int(img.size[0] / self.grid_size)
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        #img_np = np.array(img)
        #print(img_np)
        return self._image_transformer(img)

    def return_order(self, order):
        if order == 0:
            return 0
        else:
            return order

    def __getitem__(self, index):
        img = self.get_image(index)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        # order = 0
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:  # original image, no shuffle
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        data = torchvision.utils.make_grid(data, self.grid_size, padding=0)
        img = self._image_tensor(img)
        result = torch.cat((data, img), 2)
        return result, int(order)

    def __len__(self):
        return len(self.names)

    #def __retrieve_permutations(self, classes):
    #    all_perm = np.load('./permutations/permutations_hamming_max_4_%d.npy' % (classes))
    #    # from range [1,9] to [0,8]
    #    if all_perm.min() == 1:
    #        all_perm = all_perm - 1
    #    return all_perm


class Jigsaw_Loader(data.Dataset):
    def __init__(self, data_root, train=True):
        self.root = data_root
        # self.img_size = (256, 256)
        self.files = collections.defaultdict(list)

        self.train = train

        if self.train:
            split = "train"
            self.split = split
            fullpath = os.path.join(self.root, split + '.txt')
            if os.path.isfile(fullpath):
                print('Loading training file from file list: ', fullpath)
                file_list = tuple(open(fullpath, 'r'))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
            else:
                raise Exception("Training file list not exist: ", fullpath)
        else:
            split = "val"
            self.split = split
            fullpath = os.path.join(self.root, split + '.txt')
            if os.path.isfile(fullpath):
                print('Loading validation file from file list: ', fullpath)
                file_list = tuple(open(fullpath, 'r'))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
            else:
                raise Exception("Validation file list not exist: ", fullpath)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = os.path.join(self.root + img_name)
        img = cv2.imread(img_path)
        txt_name = self.root + str(img_name) + '.txt'
        f_txt = open(txt_name, mode='r')
        patch_order = f_txt.readlines()
        f_txt.close()

        transform = []
        # if mode == 'train':
        #     transform.append(T.RandomHorizontalFlip())
        # transform.append(T.CenterCrop(crop_size))
        # transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        return {'image': transform(img), 'patch_order': patch_order}