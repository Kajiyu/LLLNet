#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

mp3s = glob.glob("/data/mp3/videos/*/*/*/*.mp4.mp3")
print len(mp3s)

pbar = tqdm(total=len(mp3s))
for mp3path in mp3s:
    try:
        y, sr = librosa.load(mp3path)
    except:
        continue
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    my_dpi = 100.
    fig = plt.figure(figsize=(287./my_dpi, 229./my_dpi), dpi=my_dpi)
    librosa.display.specshow(log_S, sr=sr, cmap='gray_r')
    plt.tight_layout()
    specpath = "/data/mp3_spec/" + mp3path.split("/mp3/")[1].split(".mp3")[0] + ".jpg"
    plt.savefig(specpath, bbox_inches="tight", pad_inches=0.0)
    pbar.update(1)
    plt.clf()
    plt.close(fig)
pbar.close()
print "finished."
