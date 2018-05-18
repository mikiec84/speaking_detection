# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

SHAPE_PREDICTOR_PATH = 'data/shape_predictor_68_face_landmarks.dat'

# initialize dlib's face dlib_find_faces (HOG-based) and then create
# the facial landmark dlib_face_landmarks
print("[INFO] loading facial landmark dlib_face_landmarks...")
dlib_find_faces = dlib.get_frontal_face_detector()
dlib_face_landmarks = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=False).start()

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale frame
    people = dlib_find_faces(frame, 0)
    
    # loop over the face detections
    for i, rect in enumerate(people):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = dlib_face_landmarks(frame, rect)
        shape = face_utils.shape_to_np(shape)
 
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
        # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
      
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the Esc or `q` key was pressed, break from the loop
    if key == ord("q") or key == 27:
        break
    
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

