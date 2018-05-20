import cv2


class FileVideoStream:

    def __init__(self, path):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

    def read(self):
        (grabbed, frame) = self.stream.read()

        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            return None

        return frame

    def stop(self):
        pass