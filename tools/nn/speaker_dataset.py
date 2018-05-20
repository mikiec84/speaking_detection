import os.path

import skimage.io
import tqdm

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
                clsbase = 0 if sentence.startswith('head') else 1
                diritems = os.listdir(vpath + '/' + sentence)
                dirlast = len(diritems) - 8
                for seq in sorted(diritems):
                    cls = 0 if (int(seq) < 12 or int(seq) >= dirlast) else clsbase
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

