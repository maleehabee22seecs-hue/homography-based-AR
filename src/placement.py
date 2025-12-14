import cv2
import numpy as np

class PlacementController:
    def __init__(self, initial_frame, preview_size=120):
        self.initial_frame = initial_frame.copy()
        self.preview_size = preview_size
        self.anchor = None
        self.locked = False
        self.roi_corners = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.locked:
            self.anchor = (x, y)
            self._compute_roi()
            print("Anchor selected at:", self.anchor)

    def _compute_roi(self):
        x, y = self.anchor
        r = self.preview_size

        tl = (x - r, y - r)
        tr = (x + r, y - r)
        br = (x + r, y + r)
        bl = (x - r, y + r)

        self.roi_corners = np.array([tl, tr, br, bl], dtype=np.float32)

    def draw_preview(self, frame):
        if self.roi_corners is None:
            return frame
        
        f = frame.copy()
        pts = self.roi_corners.astype(int)
        cv2.polylines(f, [pts], True, (0,255,0), 2)
        cv2.circle(f, self.anchor, 5, (0,0,255), -1)
        return f

    def lock(self):
        if self.anchor is None:
            print("No anchor to lock.")
            return
        self.locked = True
        print("Placement locked.")
