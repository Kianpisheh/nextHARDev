from torch import nn


class EventModel(nn.Module):
    def __init__(self):
        super(EventModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 20, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(20 * 128, 500)
        self.fc2 = nn.Linear(500, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        output = self.fc2(x)

        return output
