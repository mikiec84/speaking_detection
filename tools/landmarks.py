# import the necessary packages
import imutils
import cv2
from algo.tracker import Tracker
from imutils.video.webcamvideostream import WebcamVideoStream


def main():
    # initialize the video stream and allow the cammera sensor to warmup
    vs = WebcamVideoStream().start()
    # loop over the frames from the video stream

    cv2.startWindowThread()
    cv2.namedWindow('Movie', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Movie', 480, 480)
    cv2.moveWindow('Movie', 1920 + 1400, 600)

    fid = 0
    tracker = Tracker()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=480)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        tracker.update(frame)
        
        for p in tracker.people:
            p.draw(frame)
    
        # show the frame
        cv2.imshow("Movie", frame)
        fid += 1

        key = cv2.waitKey(1) & 0xFF
     
        # if the Esc or `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:
            break
        
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
