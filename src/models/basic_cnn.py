import torch.nn as nn
import torch.nn.functional as F

nclasses = 43

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1    = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool     = nn.MaxPool2d(2, 2)

        self.conv2    = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)

        self.conv3    = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)

        self.fc1      = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn   = nn.BatchNorm2d(350)

        self.fc2      = nn.Linear(350, nclasses)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        return x

    