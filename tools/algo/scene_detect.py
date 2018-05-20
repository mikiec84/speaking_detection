import dlib
import cv2
from utils.detecting import clip, dlib_c_rect
from imutils import face_utils


class SceneDetector(object):

    def __init__(self):
        self.width = 0
        self.height = 0

    def setup(self, frame):
        h, w, _c = frame.shape
        self.width = w
        self.height = h
        self.base_rect = dlib.rectangle(*map(int, [w * 0.2, h * 0.2, w * 0.8, h * 0.8]))
        self.border_rect = dlib.rectangle(*map(int, [w * 0.05, h * 0.05, w * 0.95, h * 0.95]))
        self.center_rect = dlib.rectangle(*map(int, [w * 0.4, h * 0.4, w * 0.6, h * 0.6]))
        self.scene_id = 1
        self.new_tracker(frame, False)
    
    def new_tracker(self, frame, new_scene):
        self.predicted_rect = self.base_rect
        self.tracker = dlib.correlation_tracker()
        self.tracker.start_track(frame, self.base_rect)
        self.fade = 100
        if new_scene:
            self.scene_id += 1
    
    def update(self, frame):
        if not self.width:
            self.setup(frame)
        else:
            similarity = self.tracker.update(frame, self.base_rect)
            if similarity < 7:
                self.fade -= 10
            else:
                self.fade += 10
            if self.fade <= 0:
                self.new_tracker(frame, True)
                return True
            self.fade = clip(self.fade, 0, 100)
                
            self.predicted_rect = self.tracker.get_position()
            c = self.center_rect
            b = self.border_rect
            p = self.predicted_rect
            if p.left() < b.left() or p.right() > b.right() or p.top() < b.top() or p.bottom() > b.bottom():
                self.new_tracker(frame, True)
                return True
            if c.left() < p.left() or c.right() > p.right() or c.top() < p.top() or c.bottom() > p.bottom():
                self.new_tracker(frame, False)
                return False
    
    def draw(self, frame):
        rect = dlib_c_rect(self.predicted_rect)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
     
        # show the face number
        s = "Scene #{}".format(self.scene_id, int(self.fade))
        cv2.putText(frame, s, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
