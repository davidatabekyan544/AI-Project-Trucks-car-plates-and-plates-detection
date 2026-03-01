import torch
import torch.nn as nn
from torchvision import models


class BidirectionalGRU(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, num_layers=2, dropout=0.5)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        output = self.embedding(recurrent.view(t * b, h))
        return output.view(t, b, -1)


class CRNN(nn.Module):
    def __init__(self, n_class, nh=512):
        super(CRNN, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights='DEFAULT')

        # Extract features and modify pooling to preserve Width
        self.cnn = mobilenet.features

        # We modify the last few layers of MobileNet's feature extractor
        # to ensure the output sequence is long enough for 7-8 characters.
        # MobileNetV3-Large ends with 960 channels.
        # With 64x320 input and our pooling fix, H=2, W=10 or 20.
        self.rnn = nn.Sequential(
            BidirectionalGRU(1920, nh, nh),
            BidirectionalGRU(nh, nh, n_class)
        )

    def forward(self, x):
        # x: [Batch, 3, 64, 320]
        features = self.cnn(x)  # [Batch, 960, 2, 10]

        b, c, h, w = features.size()
        # Flatten Height into Channels: 960 * 2 = 1920
        features = features.view(b, c * h, w)
        features = features.permute(2, 0, 1)  # [W, B, C]
        return self.rnn(features)