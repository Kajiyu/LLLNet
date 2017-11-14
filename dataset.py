#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random
import os, sys
from PIL import Image
import numpy as np


class LLLDataset():
    def __init__(self, spec_glob, c_mode = "rgb"):
        self.spec_paths = glob.glob(spec_glob)
        self.c_mode = c_mode
        self._counter = 0
    
    
    def create_batch(self, batch_size):
        if self._counter + batch_size >= len(self.spec_paths):
            self._counter = 0
            random.shuffle(self.spec_paths)
        batch_spec_paths = self.spec_paths[self._counter : self._counter+batch_size]
        img_batch_arr = []
        spec_batch_arr = []
        label_batch_arr = []
        for tmp_path in batch_spec_paths:
            try:
                test_imgs = glob.glob("/data/frames/" + tmp_path.split("mp3_spec/")[-1].split(".jpg")[0]+"/*.jpg")
                img = Image.open(random.choice(test_imgs))
            except:
                continue
            img_arr = np.asarray(img)
            img_arr = np.resize(img_arr, (224, 224, 3))
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1], 3)) / 255.
            img_batch_arr.append(img_arr)
            if random.random() > 0.5:
                tmp_path = random.choice(self.spec_paths)
                label_batch_arr.append([0., 1.])
            else:
                label_batch_arr.append([1., 0.])
            img = Image.open(tmp_path)
            img = img.convert('L')
            img_arr = np.asarray(img)
            img_arr = np.resize(img_arr, (199, 257))
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1], 1)) / 255.
            spec_batch_arr.append(img_arr)
        self._counter = self._counter + batch_size
        return (np.array(img_batch_arr), np.array(spec_batch_arr), np.array(label_batch_arr))