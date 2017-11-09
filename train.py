#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
from model import LLLNet


if __name__ == '__main__':
    lll_net = LLLNet()("train")