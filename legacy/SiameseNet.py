import torch.nn as nn
import torch.nn.functional as F

# --- 2. Siamese Network Architecture ---
class SiameseNet(nn.Module):
    def __init__(self, patch_size=32):
        super(SiameseNet, self).__init__()
        self.patch_size = patch_size

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        dummy_input_size = self.patch_size // (2**3)
        self.fc1_input_features = 256 * dummy_input_size * dummy_input_size

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc1_input_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2