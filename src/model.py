import os

from efficientnet_pytorch import EfficientNet
from facer.farl import load_farl
import torch
from torch import hub, nn

from utils.sam import SAM


class Detector(nn.Module):
    def __init__(self, name='efficientnet'):
        super(Detector, self).__init__()
        if name == 'efficientnet':
            self.net = EfficientNet.from_pretrained(
                "efficientnet-b4", advprop=True, num_classes=1
            )
        elif name == 'farl':
            weights_path = os.path.join(hub.get_dir(), 'checkpoints', 'FaRL-Base-Patch16-LAIONFace20M-ep64.pth')
            if not os.path.exists(weights_path):
                hub.download_url_to_file(
                    'https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth',
                    weights_path
                )
            farl = load_farl('base', weights_path)
            self.net = nn.Sequential(farl, nn.Linear(farl.output_dim, 1))
        else:
            raise ValueError(name)
        self.cel = nn.BCEWithLogitsLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(dim=1)
        return x

    def training_step(self, x, target):
        for i in range(2):
            pred_cls = self(x)
            if i == 0:
                pred_first = pred_cls
            loss_cls = self.cel(pred_cls, target)
            loss = loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return pred_first
