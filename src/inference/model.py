import os
from torch import hub
from torch import nn
from efficientnet_pytorch import EfficientNet
from facer.farl import load_farl


class Detector(nn.Module):
    def __init__(self, name='efficientnet'):
        super(Detector, self).__init__()
        self.name = name
        if name == 'efficientnet':
            self.net = EfficientNet.from_pretrained(
                "efficientnet-b4", advprop=True, num_classes=2
            )
        elif name == 'farl':
            weights_path = os.path.join(hub.get_dir(), 'checkpoints', 'FaRL-Base-Patch16-LAIONFace20M-ep64.pth')
            if not os.path.exists(weights_path):
                hub.download_url_to_file(
                    'https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth',
                    weights_path
                )
            farl = load_farl('base', weights_path)
            self.net = nn.Sequential(farl, nn.Linear(farl.output_dim, 2))
        else:
            raise ValueError(name)

    def forward(self, x):
        x = self.net(x)
        return x

