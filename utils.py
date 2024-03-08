import cv2, os
from tqdm import tqdm
import itertools, imageio, torch, random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import datasets
from scipy.misc import imresize
from torch.autograd import Variable


def retrieve_permutations(grid_size, classes):
    all_perm = np.load('./permutations/permutations_hamming_max_%d_%d.npy' % (grid_size**2, classes))
    # from range [1,9] to [0,8]
    if all_perm.min() == 1:
        all_perm = all_perm - 1
    return all_perm


def read_flows(grid_size, classes):
    all_perm = np.load('./permutations/flows_32_max_%d_%d.npy' % (grid_size**2, classes))
    return all_perm


def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def random_input4_combine(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    n = 1
    for f in tqdm(file_list):
        order_original = [[0, 0], [0, 1], [1, 0], [1, 1]]
        rgb_img = cv2.imread(os.path.join(root, f))
        print('name:%s' % f)
        rgb_img = cv2.resize(rgb_img, (256, 256))
        random.shuffle(order_original)
        # order_random = random.shuffle(order_original)
        random_rgb_img = np.zeros([256,256,3], np.uint8)
        random_rgb_img[0:128, 0:128] = rgb_img[order_original[0][0]*128:(order_original[0][0]+1)*128, order_original[0][1]*128:(order_original[0][1]+1)*128]
        random_rgb_img[128:256, 0:128] = rgb_img[order_original[1][0]*128:(order_original[1][0]+1)*128, order_original[1][1]*128:(order_original[1][1]+1)*128]
        random_rgb_img[0:128, 128:256] = rgb_img[order_original[2][0] * 128:(order_original[2][0]+1)*128, order_original[2][1] * 128:(order_original[2][1]+1) * 128]
        random_rgb_img[128:256, 128:256] = rgb_img[order_original[3][0]*128:(order_original[3][0]+1)*128, order_original[3][1]*128:(order_original[3][1]+1)*128]
        result = np.concatenate((rgb_img, random_rgb_img), 1)
        cv2.imwrite(os.path.join(save, f), result)
        print('processing img: %d, name:%s, order:%s' % (n, f, order_original))
        n += 1


def random_input9(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    n = 1
    for f in tqdm(file_list):
        order_original = [[i, j] for i in range(3) for j in range(3)]
        rgb_img = cv2.imread(os.path.join(root, f))
        print('name:%s' % f)
        rgb_img = cv2.resize(rgb_img, (300, 300))
        random.shuffle(order_original)
        random_rgb_img = np.zeros([300, 300, 3], np.uint8)
        num = 0
        for i in range(3):
            for j in range(3):
                random_rgb_img[i*100:(i+1)*100, j*100:(j+1)*100] = rgb_img[order_original[num][0]*100:(order_original[num][0]+1)*100,order_original[num][1]*100:(order_original[num][1]+1)*100]
                num += 1
        cv2.imwrite(os.path.join(save, f), random_rgb_img)
        result = np.concatenate((random_rgb_img, rgb_img), 1)
        cv2.imwrite(os.path.join(save, f), result)
        print('processing img: %d, name:%s, order:%s' % (n, f, order_original))
        n += 1


def random_input9_combine(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    import pdb
    pdb.set_trace()
    n = 1
    for f in tqdm(file_list):
        order_original = [[i, j] for i in range(3) for j in range(3)]
        rgb_img = cv2.imread(os.path.join(root, f))
        print('name:%s' % f)
        rgb_img = cv2.resize(rgb_img, (300, 300))
        random.shuffle(order_original)
        random_rgb_img = np.zeros([300, 300, 3], np.uint8)
        num = 0
        for i in range(3):
            for j in range(3):
                random_rgb_img[i*100:(i+1)*100, j*100:(j+1)*100] = rgb_img[order_original[num][0]*100:(order_original[num][0]+1)*100,order_original[num][1]*100:(order_original[num][1]+1)*100]
                num += 1
        result = np.concatenate((random_rgb_img, rgb_img), 1)
        cv2.imwrite(os.path.join(save, f), result)
        f_txt = open(save + '/' + '%s.txt' % f, mode='w')
        f_txt.write('%s' % order_original[0])
        f_txt.close()
        print('processing img: %d, name:%s, order:%s' % (n, f, order_original))
        n += 1


def edge_promoting(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    n = 1
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f))
        rgb_img = cv2.resize(rgb_img, (256, 256))
        gauss_img = cv2.GaussianBlur(rgb_img, (5, 5), 2.5)

        result = np.concatenate((rgb_img, gauss_img), 1)

        cv2.imwrite(os.path.join(save, str(n) + '.png'), result)
        #print('processing img: %d' %n)
        n += 1


