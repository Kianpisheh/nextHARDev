import os

import tensorflow_hub as hub


import librosa


# Load the model.
vggish = hub.load(
    "https://kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1"
)


audio_path = "./augmented"

audio_files = os.listdir(audio_path)

for audio_file in audio_files:
    embeddings, sr = librosa.load(os.path.join(audio_path, audio_file), sr=None)
    embeddings = librosa.to_mono(embeddings)
    embeddings = librosa.resample(embeddings, orig_sr=sr, target_sr=16000)

    embeddings.shape.assert_is_compatible_with([None, 128])
