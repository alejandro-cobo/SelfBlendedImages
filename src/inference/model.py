from torch import nn
from efficientnet_pytorch import EfficientNet
from ..farl import FARL_PRETRAIN_PATH, load_farl


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

    def forward(self, x):
        x = self.net(x)
        return x
