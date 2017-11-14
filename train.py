#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
from model import LLLNet
from dataset import LLLDataset


if __name__ == '__main__':
    lll_net = LLLNet()("train")
    dataset = LLLDataset("/data/mp3_spec/videos/*/*/*/*.jpg")