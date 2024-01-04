import os
import pickle

import tensorflow_hub as hub


import librosa


# Load the model.
vggish = hub.load(
    "https://kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1"
)


features = {}
audio_path = "./data/test"
for audio_file in os.listdir(audio_path):
    audio_wav, sr = librosa.load(os.path.join(audio_path, audio_file), sr=None)
    audio_wav = librosa.to_mono(audio_wav)
    audio_wav = librosa.resample(audio_wav, orig_sr=sr, target_sr=16000)

    embeddings = vggish(audio_wav)

    embeddings.shape.assert_is_compatible_with([None, 128])
    embeddings = embeddings.numpy()

    # get feature label
    label = audio_file.split("_")[0]
    feature_list = [row.tolist() for row in embeddings]
    if label in features:
        features[label].extend(feature_list)
    else:
        features[label] = feature_list


# save features
with open(os.path.join(audio_path, "test.pkl"), "wb") as f:
    pickle.dump(features, f)
