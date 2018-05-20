import cv2
from utils.detecting import dlib_c_rect
from imutils import face_utils


class VoiceDetector(object):

    def __init__(self):
        self.vol = 0
        self.voice_on = False

    def update(self, sound_chunk):
        #self.vol = max(self.vol, sound) * (1-0.2/50)
        self.voice_on = True
    
    def draw(self, frame):
        rect = dlib_c_rect(self.predicted_rect)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # show the face number
        s = "Scene #{}".format(self.scene_id, int(self.fade))
        if self.voice_on:
            cv2.putText(frame, s, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
