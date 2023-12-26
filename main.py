import os
import re

import torch
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import resampy
import librosa

from vggish.vggish_input import wavfile_to_examples
from DataManager import get_action

# Load the model.
model3 = hub.load(
    "https://kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1"
)

x_wav, sr = librosa.load("./audioData/coffee-machine.wav", sr=None)
x_wav = librosa.to_mono(x_wav)
y_low = librosa.resample(x_wav, orig_sr=sr, target_sr=16000)

model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.eval()

audio_files = os.listdir("./audioData")
