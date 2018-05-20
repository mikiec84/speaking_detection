import cv2

from algo.face_tracker import FaceTracker
from algo.scene_detect import SceneDetector
from nn.speaker import Net, get_speaking_detector_final
from torchvision import transforms
import numpy
Net.__path__ = '__main__'
import os

device = os.environ.get('DEVICE', 'cpu')
device_id = int(os.environ.get('DEVICE_ID', '0'))

resizer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(300),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

spd = get_speaking_detector_final()
spd = spd.eval()
if device == 'cuda':
    spd = spd.cuda(device_id)
else:
    spd = spd.cpu()

def predict_speaking(im):
    #import torch
    #torch.save(im, '../data/tmp/im.pt')
    t = resizer(im)
    t = t[:,:224,:224]
    t = t.unsqueeze(0).float()
    if device == 'cuda':
        t = t.cuda(device_id)
    else:
        t = t.cpu()
    #torch.save(t, '../data/tmp/123.pt')
#     import matplotlib
#     matplotlib.use('agg')
#     import matplotlib.pyplot as plt
#     plt.imshow(im.permute(1,2,0))
    r = spd(t)
    #print(r[0,1].item())
    return r.argmax().float().item()
    #return (r[0,1]>1).item()


class Processor:

    def __init__(self):
        self.tracker = FaceTracker()
        self.scene = SceneDetector()

    def update(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.tracker.update(rgb)
        self.scene.update(rgb)
        
        for p in self.tracker.people:
            img = p.get_image(rgb)
            
            p.last_speaking.append(predict_speaking(img))
            p.last_speaking.append(predict_speaking(img))
            p.last_speaking = p.last_speaking[-20:]
            #print(p.last_speaking[-20:])
            p.speaking = numpy.average(p.last_speaking[-20:])
            #print(p.speaking)
        
        for p in self.tracker.people:
            p.draw(frame)

        self.scene.draw(frame)    
    
    def finish(self):
        pass
    
