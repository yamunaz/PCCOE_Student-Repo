import cv2
import time


class VideoStream:
    def __init__(self, source, reconnect_delay=2):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.connect()

    def connect(self):
        print(f"[INFO] Connecting to video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print("[ERROR] Failed to open stream. Retrying...")
            self.cap.release()
            self.cap = None

    def frames(self):
        while True:
            # If disconnected, reconnect
            if self.cap is None or not self.cap.isOpened():
                print("[WARNING] Stream disconnected. Reconnecting...")
                time.sleep(self.reconnect_delay)
                self.connect()
                continue

            ret, frame = self.cap.read()

            # If failed to read frame, reconnect
            if not ret:
                print("[WARNING] Frame read failed. Reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue

            yield frame

    def release(self):
        if self.cap:
            self.cap.release()
            print("[INFO] Stream released.")
