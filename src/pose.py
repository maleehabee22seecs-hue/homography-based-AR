import cv2
import numpy as np

def decompose_homography(H, K, prev_R=None):
    # Use OpenCV's robust decomposition
    # Returns 4 possible solutions (num, Rs, Ts, Ns)
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    
    # Solution Selection Heuristic:
    # 1. Plane normal alignment with Camera Z (assuming surface is roughly facing camera)
    # 2. Consistency with previous rotation (if available) to prevent flipping
    
    best_idx = 0
    best_score = -float('inf')
    
    for i in range(num):
        n = Ns[i].flatten()
        R = Rs[i]
        t = Ts[i].flatten()
        
        # Score 1: Normal alignment with Z (0,0,1)
        score_z = abs(n[2])
        
        # Score 2: Positive Depth Check
        # If we transform a point on the plane (e.g. 0,0,1 or 0,0,0) with this R,t, does it land in front?
        # Standard H decomp: x_cam = R * x_plane + t * d
        # Let's test the center of the plane (0,0,0) or (0,0,1).
        # Actually t itself is the position of the plane origin in cam frame (scaled by d).
        # So we really just want t[2] to be POSITIVE (in front of camera) OR
        # if the normal implies we're looking AT the plane.
        
        # In most AR cases, the plane is in front of the camera.
        # But cv2.decomposeHomographyMat returns t normalized by d (depth of plane).
        # If d is positive (plane in front), then t[2] should roughly generally be positive.
        
        # Let's enforce that the camera is on the side of the plane defined by normal?
        # Actually simpler: standard decompositoin returns 4 solutions:
        # 2 are "in front", 2 are "behind".
        # We MUST pick one where the point is in front.
        
        # Check Z component of translation (scaled).
        # Ideally t[2] should be positive if the origin of the plane is in front.
        # But this depends on where the origin is.
        # Let's assume the tracked feature centroid is roughly at (0,0,0) of the plane coord system.
        
        is_in_front = 1.0 if t[2] > 0 else -1.0
        
        score_consistency = 0.0
        if prev_R is not None:
             score_consistency = np.trace(R @ prev_R.T)
        
        # Weighted combination
        # Huge penalty for being behind camera
        if prev_R is not None:
            total_score = score_z + 2.0 * score_consistency + 5.0 * is_in_front
        else:
            total_score = score_z + 5.0 * is_in_front

        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return Rs[best_idx], Ts[best_idx]

class PoseFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.R_curr = np.eye(3)
        self.t_curr = np.zeros((3,1))
        self.initialized = False
        
    def update(self, R, t):
        # Check for NaNs or Infs
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            return self.R_curr, self.t_curr

        if not self.initialized:
            self.R_curr = R
            self.t_curr = t
            self.initialized = True
            return R, t
            
        # Simple Linear blending for R 
        R_new = self.alpha * R + (1 - self.alpha) * self.R_curr
        
        # Re-orthonormalize Rotation Matrix using SVD
        try:
            U, _, Vt = np.linalg.svd(R_new)
            self.R_curr = U @ Vt
        except np.linalg.LinAlgError:
            # Fallback if SVD fails: keep previous rotation
            pass
        
        self.t_curr = self.alpha * t + (1 - self.alpha) * self.t_curr
        
        return self.R_curr, self.t_curr

    def reset(self):
        self.initialized = False
