import cv2
import numpy as np


class LocalVisionModel:
    """
    Upgraded Local Vision Model

    Detects:
    - Fire risk (HSV fire mask)
    - Crash detected (Optical Flow motion spike + frame diff)
    - Severity estimation
    """

    def __init__(self):
        self.prev_gray = None
        self.motion_history = []

    def detect_fire(self, frame):
        """
        Fire detection using HSV orange/yellow/red detection.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_fire = np.array([0, 60, 140])
        upper_fire = np.array([40, 255, 255])

        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        fire_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]

        fire_ratio = fire_pixels / total_pixels

        return fire_ratio > 0.008  # reduced threshold (more sensitive)

    def motion_spike_frame_diff(self, gray):
        """
        Detect crash using frame difference.
        """
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        motion_pixels = cv2.countNonZero(thresh)

        self.prev_gray = gray
        return motion_pixels

    def motion_spike_optical_flow(self, prev_gray, gray):
        """
        Detect crash using optical flow magnitude.
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)

        return avg_mag

    def detect_crash(self, gray):
        """
        Combined crash detector:
        - Frame diff spike
        - Optical flow spike
        """

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        diff_motion = self.motion_spike_frame_diff(gray)
        flow_motion = self.motion_spike_optical_flow(self.prev_gray, gray)

        # store history
        self.motion_history.append((diff_motion, flow_motion))
        if len(self.motion_history) > 15:
            self.motion_history.pop(0)

        # update prev_gray
        self.prev_gray = gray

        # not enough history
        if len(self.motion_history) < 6:
            return False

        # compute averages excluding current frame
        prev_diff_avg = np.mean([m[0] for m in self.motion_history[:-1]])
        prev_flow_avg = np.mean([m[1] for m in self.motion_history[:-1]])

        current_diff = self.motion_history[-1][0]
        current_flow = self.motion_history[-1][1]

        # crash if big spike happens
        diff_spike = prev_diff_avg > 0 and current_diff > prev_diff_avg * 1.8
        flow_spike = prev_flow_avg > 0 and current_flow > prev_flow_avg * 1.6

        # additional low-level fallback (bike crash small motion)
        if current_diff > 12000:
            return True

        if diff_spike or flow_spike:
            return True

        return False

    def analyze(self, frame):
        """
        Returns structured scene dictionary.
        """

        # resize frame for stable detection
        frame_small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        fire_detected = self.detect_fire(frame_small)
        crash_detected = self.detect_crash(gray)

        # default outputs
        severity = "low"
        injuries = "no"
        road_blocked = "no"

        # crash = medium severity
        if crash_detected:
            severity = "medium"
            injuries = "yes"

        # fire = high severity
        if fire_detected:
            severity = "high"

        # crash + fire = very dangerous
        if crash_detected and fire_detected:
            severity = "high"
            injuries = "yes"
            road_blocked = "yes"

        scene = {
            "severity": severity,
            "injuries": injuries,
            "fire_risk": "yes" if fire_detected else "no",
            "road_blocked": road_blocked,
            "crash_detected": "yes" if crash_detected else "no"
        }

        return scene
