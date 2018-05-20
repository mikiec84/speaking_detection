import os

import skimage.io
from torch.nn import Module
import torch.nn
from torchvision.models import resnet18
from nn.speaker_dataset import Dataset  # @UnusedImport

os.environ['TORCH_MODEL_ZOO'] = '../data/'

VIDTIMIT_PATH = '../data/vidtimit/'

skimage.io.use_plugin('pil')


class Net(Module):

    def __init__(self):
        super().__init__()

        resnet = resnet18(pretrained=True)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 2)
        )
        # print(len(list(self.features.parameters())))
        for p in list(self.features.parameters())[:20]:
            p.requires_grad = False
    
    def forward(self, x, **kw):
        # X = F.softmax(self.basenet(X))
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def get_speaking_detector_final():
    m = torch.load('../data/speaker.pt')
    m = m.eval();
    return m


def get_speaking_detector(e):
    m = torch.load('../data/speaker/model.e{}.pt'.format(e))
    m = m.eval();
    return m
