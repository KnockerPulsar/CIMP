# Borrowed from: https://stackoverflow.com/a/55131847

from threading import Thread
import cv2


class ThreadedVideoStream(object):
    """
    Runs webcam capture asyncronously.

    Helps with increasing responsiveness since it doesn't block the main thread.
    """

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while self.running:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def release(self):
        self.running = False
        self.capture.release()

    def get_frame(self):
        try:
            return self.status, self.frame
        except:
            return False, None
