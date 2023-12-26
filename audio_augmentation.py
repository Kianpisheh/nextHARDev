import os

import librosa
import soundfile as sf
from audiomentations import (
    Compose,
    AddGaussianNoise,
    PitchShift,
    Shift,
    TimeStretch,
    Gain,
)


augment = Compose(
    [
        AddGaussianNoise(min_amplitude=0.015, max_amplitude=0.03, p=0.4),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.25),
        Gain(min_gain_db=-8, max_gain_db=8, p=0.5),
    ],
    shuffle=True,
)


source_path = "./audioData"
audio_files = os.listdir(source_path)
N = 3  # num of augmentations
for audio_file in audio_files:
    source_wav, sr = librosa.load(os.path.join(source_path, audio_file), sr=None)
    for i in range(N):
        augmented_wav = augment(source_wav, sr)
        sf.write(
            f"{os.path.join(source_path, audio_file.split('.')[0])}_{i}.wav",
            augmented_wav,
            sr,
        )
