import cv2

from algo.face_tracker import FaceTracker
from algo.scene_detect import SceneDetector
from nn.speaker import Net, get_speaking_detector_final
from torchvision import transforms
Net.__path__ = '__main__'
import os

device = os.environ.get('DEVICE', 'cpu')
device_id = int(os.environ.get('DEVICE_ID', '0'))

resizer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

spd = get_speaking_detector_final()
spd = spd.eval()
if device == 'cuda':
    spd = spd.cuda(device_id)
else:
    spd = spd.cpu()

def predict_speaking(im):
    t = resizer(im.transpose(2,1,0))
    t = t[:,:224,:224]
    t = t.unsqueeze(0).float()
    if device == 'cuda':
        t = t.cuda(device_id)
    else:
        t = t.cpu()

#     import matplotlib
#     matplotlib.use('agg')
#     import matplotlib.pyplot as plt
#     plt.imshow(im.permute(1,2,0))
    r = spd(t)
    print(r)
    return r.argmax().item()


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
            p.speaking = predict_speaking(img)
        
        for p in self.tracker.people:
            p.draw(frame)

        self.scene.draw(frame)    
    
    def finish(self):
        pass
    
