import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from farl import FARL_PRETRAIN_PATH, load_farl
from utils.sam import SAM


class Detector(nn.Module):
    def __init__(self, name='efficientnet'):
        super(Detector, self).__init__()
        self.name = name
        if name == 'efficientnet':
            self.net = EfficientNet.from_pretrained(
                "efficientnet-b4", advprop=True, num_classes=2
            )
        elif name == 'farl':
            farl = load_farl('base', FARL_PRETRAIN_PATH)
            self.net = nn.Sequential(farl, nn.Linear(farl.output_dim, 2))
        else:
            raise ValueError(name)
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.net(x)
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
