import torch

from vggish.vggish_input import wavfile_to_examples
from DataManager import get_action


get_action("")

aud = wavfile_to_examples("./test.wav")

model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.eval()
g = model.forward("./test.wav")
x = 1
