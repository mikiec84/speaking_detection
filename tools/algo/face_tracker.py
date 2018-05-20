from algo.face import Face
import dlib

# dlib's face dlib_find_faces (HOG-based)
dlib_find_faces = dlib.get_frontal_face_detector()


class FaceTracker(object):

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
            self.people.append(Face(frame, rect))
