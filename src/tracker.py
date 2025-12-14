import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, max_corners=200):
        self.initialized = False
        self.max_corners = max_corners
        self.lk_params = dict(winSize=(21,21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # Replenishment params
        self.min_pts = 50 
        self.detect_interval = 5 
        
        # State
        self.prev_gray = None
        self.prev_pts = None # Points in previous frame
        self.initial_pts = None # Points in the very first frame (corresponding to prev_pts)

    def initialize(self, frame, roi_corners):
        """
        frame: BGR image of the initial frame
        roi_corners: 4x2 float32 polygon in initial frame coordinates
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(mask, roi_corners.astype(int), 255)

        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
                                      qualityLevel=0.01, minDistance=4, mask=mask)
        if pts is None:
            print("FeatureTracker.initialize(): no corners found in ROI")
            self.initialized = False
            return

        # Initialize state
        self.prev_gray = gray.copy()
        # Shape: (N, 1, 2)
        self.prev_pts = pts.copy() 
        self.initial_pts = pts.copy() # Keep the original coordinates of these points
        
        self.initialized = True
        print(f"FeatureTracker initialized with {len(self.prev_pts)} points")

    def track(self, curr_frame):
        """
        Track features. Returns H from Initial Frame -> Current Frame.
        """
        if not self.initialized:
            return None, None, None

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Track from prev_gray -> curr_gray
        p0 = self.prev_pts
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, **self.lk_params)

        if p1 is None:
            return None, None, None

        st = st.reshape(-1)
        
        # Select good points in current frame
        good_curr = p1[st==1].reshape(-1, 1, 2)
        
        # Also keep corresponding points from initial frame (so we can compute H_0->t)
        good_init = self.initial_pts[st==1].reshape(-1, 1, 2)

        # Update internal state (for next incremental track)
        self.prev_gray = curr_gray
        self.prev_pts = good_curr
        self.initial_pts = good_init # Narrow down the set of tracked points

        if len(good_curr) < 4:
            # Lost tracking
            return None, good_init, good_curr

        # Compute Homography from Initial Pts -> Current Pts
        # This gives us the transform relative to the start (absolute pose)
        H, mask_hom = cv2.findHomography(good_init, good_curr, cv2.RANSAC, 4.0)
        
        # Feature Replenishment
        if H is not None and len(good_curr) < self.min_pts:
             good_curr, good_init = self.replenish_features(curr_gray, good_curr, H, good_init)
             # Update state with replenished points
             self.prev_pts = good_curr
             self.initial_pts = good_init

        return H, good_init, good_curr

    def replenish_features(self, curr_gray, curr_pts, H, initial_pts):
        """
        Find new features in curr_gray (inside mask of curr_pts) 
        and map them back to initial frame using H_inv.
        """
        if len(curr_pts) < 4:
            return curr_pts, initial_pts
            
        mask = np.zeros_like(curr_gray, dtype=np.uint8)
        # Dilate hull slightly to cover object
        hull = cv2.convexHull(curr_pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Detect NEW points
        new_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=self.max_corners,
                                          qualityLevel=0.01, minDistance=5, mask=mask)
        
        if new_pts is None:
            return curr_pts, initial_pts
            
        # Filter points that are too close to existing points
        # (This is a simple concatenation for now, sophisticated filtering is better but complex)
        
        # Transform detected points back to initial frame: P_init = H_inv * P_curr
        # H is Init -> Curr, so we need H_inv
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return curr_pts, initial_pts
            
        # Reshape new_pts to (N, 1, 2)
        N = new_pts.shape[0]
        pts_hc = np.ones((3, N))
        pts_hc[:2, :] = new_pts.reshape(N, 2).T
        
        pts_init_hc = H_inv @ pts_hc
        pts_init_hc /= pts_init_hc[2, :] # Normalize z
        
        new_init_pts = pts_init_hc[:2, :].T.reshape(N, 1, 2)
        
        # Merge
        updated_curr = np.concatenate((curr_pts, new_pts), axis=0)
        updated_init = np.concatenate((initial_pts, new_init_pts), axis=0)
        
        print(f"[Tracker] Replenished: {len(curr_pts)} -> {len(updated_curr)} points")
        return updated_curr, updated_init
