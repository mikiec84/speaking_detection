import cv2

from algo.face_tracker import FaceTracker
from algo.scene_detect import SceneDetector
import nn.speaker
import os

device = os.environ.get('DEVICE', 'cpu')
device_id = int(os.environ.get('DEVICE_ID', '0'))

class Processor:

    def __init__(self):
        self.tracker = FaceTracker()
        self.scene = SceneDetector()
        self.speaking_net = nn.speaker.get_speaking_detector(12)

    def update(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.tracker.update(rgb)
        self.scene.update(rgb)
        
        
        for p in self.tracker.people:
            p.draw(frame)

        self.scene.draw(frame)    

    
    def finish(self):
        pass
    
    
