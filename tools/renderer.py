# import the necessary packages
import sys
import cv2
from algo.processor import Processor
import imutils
from imutils.video.webcamvideostream import WebcamVideoStream
from utils.fvs import FileVideoStream

# one don't talk then one talks:
# data/lilir/twotalk/clip_2thWiYmVlo.avi
def render(source):
    # initialize the video stream and allow the cammera sensor to warmup
    if source:
        vs = FileVideoStream(source)
        if not vs.stream.isOpened():
            print("File couldn't be opened")
            sys.exit()
        video_fps = vs.stream.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        ofn = '../data/output/{}'.format(sys.argv[1].split('/')[-1].replace('clip_', 'output_'))
        out = cv2.VideoWriter(ofn, fourcc, video_fps, (640, 512))
    else:
        vs = WebcamVideoStream().start()
        out = None
    # If Camera Device is not opened, exit the program
    # loop over the frames from the video stream

    cv2.startWindowThread()
    cv2.namedWindow('Movie', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Movie', 640+60, 480+60)
    cv2.moveWindow('Movie', 1920 + 1400, 600)

    p = Processor()

    video_fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print("Video FPS: {0}".format(video_fps))

    while True:
        frame = vs.read()
        if frame is None:
            break
        #print("Read:", frame.shape)
        frame = imutils.resize(frame, width=640)
        
        p.update(frame)        
        # show the frame
        #print(frame.shape)
        cv2.imshow("Movie", frame)
        if out:
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
     
        # if the Esc or `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:
            break
    
    print("Finished")
    vs.stop()
    vs.stream.release()
    
    if out:
        out.release()
    
    p.finish()
    # do a bit of cleanup
    cv2.destroyWindow("Movie")
    

if __name__ == '__main__':
    if sys.argv[1:]:
        render(sys.argv[1]) # 'data/lilir/twotalk/clip_25hCrqBFGn.avi'
    else:
        #webcam
        render(0)
