import os

# get the event action
# extract audio waveform
# get VGGish embeddings
# return embeddings with labels


def get_action(action: str):
    data_abs_path = "/media/kian/889A3B4B9A3B3552/Epic_Kitchen_100"
    files = os.listdir(data_abs_path)
    print(files)
