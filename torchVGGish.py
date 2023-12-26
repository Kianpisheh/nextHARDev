import torch
import librosa

model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.postprocess = True
model.eval()


x1, sr = librosa.load("./audioData/chopping01.wav")
x1 = librosa.to_mono(x1)
x1 = librosa.resample(x1, orig_sr=sr, target_sr=16000)

x2, sr = librosa.load("./audioData/chopping03.wav")
x2 = librosa.to_mono(x2)
x2 = librosa.resample(x2, orig_sr=sr, target_sr=16000)


x3, sr = librosa.load("./audioData/microwave-beep01.wav")
x3 = librosa.to_mono(x3)
x3 = librosa.resample(x3, orig_sr=sr, target_sr=16000)


x4, sr = librosa.load("./audioData/microwave-beep02.wav")
x4 = librosa.to_mono(x4)
x4 = librosa.resample(x4, orig_sr=sr, target_sr=16000)


x5, sr = librosa.load("./audioData/coffee-machine01.wav")
x5 = librosa.to_mono(x5)
x5 = librosa.resample(x5, orig_sr=sr, target_sr=16000)


x6, sr = librosa.load("./audioData/coffee-machine02.wav")
x6 = librosa.to_mono(x6)
x6 = librosa.resample(x6, orig_sr=sr, target_sr=16000)

d1 = model(x1, 16000).cpu().detach().numpy().reshape(-1)
d2 = model(x2, 16000).cpu().detach().numpy().reshape(-1)
d3 = model(x3, 16000).cpu().detach().numpy().reshape(-1)
d4 = model(x4, 16000).cpu().detach().numpy().reshape(-1)
d5 = model(x5, 16000).cpu().detach().numpy().reshape(-1)
d6 = model(x6, 16000).cpu().detach().numpy().reshape(-1)

import numpy as np


def cos_sim(d1, d2):
    return np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))


s1 = cos_sim(d1, d2)
s2 = cos_sim(d1, d3)
s3 = cos_sim(d1, d4)
s4 = cos_sim(d3, d2)
s5 = cos_sim(d3, d4)
s6 = cos_sim(d1, d6)
s7 = cos_sim(d3, d6)
s8 = cos_sim(d5, d6)
print(s1, s2, s3, s4, s5, s6, s7, s8)
