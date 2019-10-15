import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .generic_model import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from ..logger import Logger

log = Logger()

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(model):

    def __init__(self, config, num_classes=1000):
        super(AlexNet, self).__init__()

        n_channels = config['n_channels']
        num_classes = config['n_classes']
        lr = config['lr']
        opt = config['optimizer']
        lossf = config['lossf']
        self.PreTrained = config['PreTrained']
        self.config = config

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if opt == 'Adam':
            self.opt = optim.Adam(self.parameters(), lr=lr)

        if lossf == 'CEL':
            self.lossf = torch.nn.CrossEntropyLoss()
        elif lossf == 'MSE':
            self.lossf = torch.nn.MSELoss()

    def loss(self, pred, target, info):

        b, ch = pred.size()

        pred = pred.contiguous().view(b, ch)
        target = target.contiguous().view(b)

        loss_c = self.lossf(pred, target)

        if info['Keep_log']:
            info['Acc'].append(self.accuracy(pred, target).data.cpu().numpy())
            info['Loss'].append(loss_c.data.cpu().numpy())

        else:
            info['Acc'] = (self.accuracy(pred, target).data.cpu().numpy() + info['Count']*info['Acc'])/(info['Count']+1)
            info['Loss'] = (loss_c.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)

        info['Count'] += 1
        return loss_c

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
