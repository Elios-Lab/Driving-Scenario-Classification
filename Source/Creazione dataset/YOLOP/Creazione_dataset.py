import time
from PIL import Image
import sys
print(sys.version, sys.platform, sys.executable)

from torch.utils.data import dataloader

import torch
import torchvision
import torchvision.transforms as transforms


import math
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import *
import pylab

import os
from os import listdir
from os.path import isfile, join, isdir
import gc

from pathlib import Path

import cv2
import numpy as np

from tqdm import tqdm


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:16:01 2021

@author: utente1
"""

import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    if half:
        model.half()  # to FP16
    
    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()
    
    class Creazione_VideoDataset():
    
        def __init__(self, directory, mode='Day', clip_len=8, num_sec=8):
            folder = Path(directory)
            self.mode = mode
            self.clip_len = clip_len-1
            self.num_sec=num_sec
            self.origin_path = []
            y = []
            # obtain all the filenames of files inside all the class folders 
            # going through each class folder one at a time
            self.fnames, labels = [], []
            for label in sorted(os.listdir(folder)):
                for intermezzo in sorted(os.listdir(os.path.join(folder,label))):
                    self.origin_path.append(os.path.join(folder, label, intermezzo))
                    
                    y.append(label)
                    for fname in os.listdir(os.path.join(folder, label, intermezzo)):
                        self.fnames.append([os.path.join(folder, label, intermezzo, fname), int(intermezzo)])
                        
                        labels.append(label)
            
            # prepare a mapping between the label names (strings) and indices (ints)
            self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
            
            # convert the list of label names into an array of label indices
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        
            
            self.label2index2 = {label:index for index, label in enumerate(sorted(set(y)))} 
            
            # convert the list of label names into an array of label indices
            self.label_array2 = np.array([self.label2index[label] for label in y], dtype=int)        
            
            mode = self.mode
            count = 0
            clips = []
            clip = []
            prev_label = -1
            controllo = -1
            num_frame = 0
            num_sec = self.num_sec
            try:
                os.mkdir('Dataset'+mode+''+str((self.clip_len+1)))
            except:
                print("error")
            img = torch.zeros((1, 3, 720, 1280), device=device)  # init img
            _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            model.eval()
            for i in np.arange(len(self.origin_path)):
                for j in (np.arange(self.clip_len+1)):
                    
                    
                    try:
                        
                        os.mkdir('Dataset'+mode+''+str((self.clip_len+1))+'/'+str(self.label_array2[i]))
                    except:
                        print("error")
                    try:
                        
                        os.mkdir('Dataset'+mode+''+str((self.clip_len+1))+'/'+str(self.label_array2[i])+'/'+str(i))
                    except:
                        print("error")
                    
                    fname = self.origin_path[i]+"/"+str(self.label_array2[i])+"_"+str(j+1)+".jpg"
                    try:
                        dataset = LoadImages(fname)
                        
                        for l, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
                            img = transform(img).to(device)
                            img = img.half() if half else img.float()  # uint8 to fp16/32
                            if img.ndimension() == 3:
                                img = img.unsqueeze(0)
                            # Inference
                            det_out, da_seg_out,ll_seg_out= model(img)
                    
                        inf_out, _ = det_out
                        
                        det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
                        det=det_pred[0]
                        
                        
                        p1 = [det.numpy(), da_seg_out.numpy(), ll_seg_out.numpy()]
                        
                        
                        x = np.asarray([p1, [0]])
                        
                        try:
                            NUM_BOX = 30
                            x1 = x[0][0]
                            x1_ = np.zeros((NUM_BOX,7, 1))
                            
                            for counter in np.arange(np.shape(x1)[0]):
                                
                                x1_[counter][0, 0] = x1[counter][0]
                                x1_[counter][1, 0] = x1[counter][1]
                                x1_[counter][2, 0] = x1[counter][2]
                                x1_[counter][3, 0] = x1[counter][3]
                                x1_[counter][4, 0] = x1[counter][4]
                                x1_[counter][5, 0] = x1[counter][5]
                                x1_[counter][6, 0] = 300    
                        
                        except:
                            print("l'errore è nelle box")
    #========================================================================================================================   
                        # create an img array for the correct validation of the process segmentation --> bounding box
                        try:
                            channels = 2
                            W = 50
                            H = 50
                            x2 = x[0][1]
                            x3 = x[0][2]
                            
                            #convert from 4d to 3d the segmentation matrixs
                            x2= x2[0,:,:,:]
                            x3= x3[0,:,:,:]
                            
                            x2 = np.transpose(x2,(1,2,0))
                            x3 = np.transpose(x3,(1,2,0))
                            x2 = np.delete(x2, 0, 2)
                            x3 = np.delete(x3, 0, 2)
                            x2 = cv2.resize(x2, (W,H), interpolation = cv2.INTER_AREA)
                            x3 = cv2.resize(x3, (W,H), interpolation = cv2.INTER_AREA)
                            
                            img = np.zeros((W,H,channels))
                            img[:,:,0] = x2
                            img[:,:,1] = x3
                                   
                            p1 = [x1_, img]
                            
                            x = np.asarray([p1, [0]])                            
                        except:
                            print("error")
                        
                        np.save('Dataset'+mode+''+str((self.clip_len+1))+'/'+str(self.label_array2[i])+'/'+str(i)+'/'+str(self.label_array2[i])+'_'+str(j)+'.npy', x)
                        
                        
                        
                        
                    except:
                        print("error")
    #The launch of Creazione_VideoDataset function automatically create the
    #dataset with the number of frames specificized (mode is not an obligatory
    #parameter, add only a string to the name of dataset. It’s useful if you
    #want to create different kind of dataset (for example different weather
    #conditions).
    frames = 80
    x = Creazione_VideoDataset("datasetNight1_" + str(i), clip_len=frames, mode = "Night1")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
