import os.path

import skimage.io
from torch.nn import Module
import torch.nn
import tqdm
from torchvision.models import resnet18

os.environ['TORCH_MODEL_ZOO'] = '../data/'

VIDTIMIT_PATH = '../data/vidtimit/'

skimage.io.use_plugin('pil')


class Dataset:

    def __init__(self, root, transform=None, include=None, exclude=[]):
        self.transform = transform
        self.images = {}
        tq = tqdm.tqdm()
        self.c = {0:0, 1: 0}
        self.items = []
        for author in os.listdir(root):
            if author in exclude: continue
            if include and not author in include: continue
            vpath = os.path.join(root, author, 'video')
            for sentence in os.listdir(vpath):
                cls = 0 if sentence.startswith('head') else 1
                for seq in os.listdir(vpath + '/' + sentence):
                    fn = os.path.join(vpath, sentence, seq)
                    self.items.append((fn, cls))
                    self.c[cls] += 1
                    tq.update(1)

    def __len__(self):
        return len(self.items)
                    
    def __getitem__(self, index):
        fn, label = self.items[index]
        # print(fn, label)
        if fn in self.images:
            img = self.images[fn]
        else:
            img = skimage.io.imread(fn)
            # self.images[fn] = img
        # print(img.shape, type(img))
        if self.transform:
            img = self.transform(img)
        return (img, label)


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


def get_speaking_detector(e):
    return torch.load('../data/speaker/model.e{}.pt'.format(e))
