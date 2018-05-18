# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy
clip = numpy.clip

SHAPE_PREDICTOR_PATH = 'data/dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_PATH = 'data/dlib/dlib_face_recognition_resnet_model_v1.dat'

# initialize dlib's face dlib_find_faces (HOG-based) and then create
# the facial landmark dlib_face_landmarks
print("[INFO] loading facial landmark dlib_face_landmarks...")
dlib_find_faces = dlib.get_frontal_face_detector()
dlib_face_landmarks = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_PATH)


def center(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cx = (x + w / 2)
    cy = (y + w / 2)
    return w, h, cx, cy


def near(rect1, rect2, q):
    w1, h1, cx1, cy1 = center(rect1)
    w2, h2, cx2, cy2 = center(rect2)
    sx = (w1 + w2) / 2
    sy = (h1 + h2) / 2
    return abs(cx1 - cx2) * q < sx and abs(cy1 - cy2) * q < sy


def dlib_c_rect(r):
    return dlib.rectangle(*map(int, [r.left(), r.top(), r.right(), r.bottom()]))


uid = 1
MAX_FADE = 100

class PersonFrame:

    def __init__(self, frame, rect):
        global uid
        self.last_good_rect = rect
        self.predicted_rect = rect
        self.tracker = dlib.correlation_tracker()
        # print(rect)
        self.tracker.start_track(frame, rect)
        # face_descriptor = facerec.compute_face_descriptor(img, shape)
        self.fade = MAX_FADE
        self.uid = uid
        uid += 1

    def update_from_pic(self, frame):
        score = self.tracker.update(frame)
        # self.predicted_rect = self.tracker.get_position()
        r = self.tracker.get_position()
        self.predicted_rect = r
        if score < 10:
            self.fade -= 11 - score
        else:
            self.fade -= 1

    def update_from_rect(self, frame, rect):
        self.tracker.update(frame, rect)
        self.last_good_rect = rect
        self.predicted_rect = rect
        self.fade = clip(self.fade + 10, 0, MAX_FADE)

    def match_faces(self, rects):
        for i, rect in enumerate(rects):
            if near(self.predicted_rect, rect, 4):
                return i

        for i, rect in enumerate(rects):
            if near(self.predicted_rect, rect, 2):
                return i
        
        return -1
    
    def draw(self, frame):
        rect = dlib_c_rect(self.predicted_rect)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
        # show the face number
        s = "Face #{} fade={}".format(self.uid + 1, int(self.fade))
        cv2.putText(frame, s, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        shape = dlib_face_landmarks(frame, rect)
        shape = face_utils.shape_to_np(shape)
        
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


class Tracker(object):
    def __init__(self):
        self.people = []
        
    def update(self, frame):
        faces = dlib_find_faces(frame, 0)
        
        for p in self.people:
            m = p.match_faces(faces)
            if m >= 0:
                rect = faces.pop(m)
                p.update_from_rect(frame, rect)
            else:
                p.update_from_pic(frame)
                
        self.people = [p for p in self.people if p.fade > 0]
    
        for rect in faces:
            self.people.append(PersonFrame(frame, rect))
