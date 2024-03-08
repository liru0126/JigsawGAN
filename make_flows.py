# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:50:28 2017

@author: bbrattol
"""
import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--classes', default=2, type=int,
                    help='Number of permutations to select')
parser.add_argument('--selection', default='max', type=str,
        help='Sample selected per iteration based on hamming distance: [max] highest; [mean] average')
args = parser.parse_args()

if __name__ == "__main__":
    outname = './permutations/flows_32_%s_4_%d'%(args.selection,args.classes)

    flow_4_2 = [
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[-32, 32], [0, 0]], [[-32, 0], [0, 32]]],
        [[[0, 0], [-32, 32]], [[0, 0], [0, 0]]],
                ]

    np.save(outname, flow_4_2)

    print('file created --> '+outname)
