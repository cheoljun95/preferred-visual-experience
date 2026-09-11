from pathlib import Path
import torch
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas


COLOR_MEAN = [0.5, 0.5, 0.5]
COLOR_STD = [0.5, 0.5, 0.5]

def inverse_transform(arr):
    '''
    Inverse torch array (C*H*W) to (H*W*C) 1~255 unit8 numpy array
    '''

    arr = arr.cpu().detach().numpy()
    mean=COLOR_MEAN
    std=COLOR_STD
    mean = np.array(mean)
    std = np.array(std)
    arr = arr.transpose(1, 2, 0)
    arr = (arr * std) + mean
    arr = np.clip(arr,0,1)
    arr = (arr * 255).astype(np.uint8)

    return arr

def inverse_transform_binary(arr):
    '''
    Inverse torch array (C*H*W) to (H*W*C) 1~255 unit8 numpy array
    '''

    arr = arr.cpu().detach().numpy()
    arr = arr.transpose(1, 2, 0)
    arr = np.concatenate([arr,arr,arr],-1)
    arr = np.clip(arr,0,1)
    arr = (arr * 255).astype(np.uint8)

    return arr

def draw_map(pred,target,resize=None):
    '''
    '''

    pred = pred.cpu().detach().numpy()
    pred = (pred * 255).astype(np.uint8)

    target = target.cpu().detach().numpy()
    target = (target * 255).astype(np.uint8)

    board = np.zeros([target.shape[0], target.shape[1],3], dtype=np.uint8)
    board[:,:,0] = pred
    board[:,:,1] = target
    if resize is not None:
        board = cv2.resize(board, (resize[1],resize[0]))

    return board

def visualize_semseg(arr):
    '''
    Color segmentation predict (H*W)
    '''
    arr = arr.cpu().detach().numpy()
    color_map = [[0,   0,   0],        # None
                 [255, 183, 0],        # Buildings
                 [153, 110, 0],        # Fences
                 [255, 200, 0],        # Other/Props
                 [255, 0,   119],      # Pedestrians
                 [0,   204, 102],      # Poles
                 [153, 153, 153],      # Roadlines
                 [128, 64,  128],      # Roads
                 [244, 35,  232],      # Sidewalks
                 [162, 212, 0],        # Vegetation
                 [140, 0,   255],      # Vehicles
                 [128, 64,  0],        # Walls
                 [220, 220, 0],        # Traffic signs
                 [80,  227, 0],        # Fields
                 [30,  144, 255],      # Self
                 [117, 86,  0],        # Ground
                 [255, 0, 0],
                 [0, 255, 0],
                 [125,125,125],
                 [255, 125,125],
                 [125,255,125],
                 [125,125,255],
                 [60,125,255],
                 [125,60,255],
                 [255,60,125],
                 [255,125,60],
                 [60,255,125],
                 [125,255,60],
                ]
    
    color_map = np.array(color_map, dtype=np.uint8)
    #arr = arr.cpu().detach().numpy()
    arr = color_map[arr]

    return arr
