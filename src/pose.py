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
        # We want normal to point TOWARDS camera for a visible surface.
        # In this conv, camera looks down +Z? No, usually -Z or +Z.
        # Let's assume standard view: Normal should have large Z component.
        score_z = abs(n[2])
        
        # KEY FIX: Ensure the normal points roughly towards the camera.
        # If n[2] is negative, it might be facing away depending on coord sys.
        # We prefer solutions where t[2] is positive (object in front).
        
        is_in_front = 1.0 if t[2] > 0 else -10.0 # Heavy penalty if behind
        
        score_consistency = 0.0
        if prev_R is not None:
             # Dot product of rotation vectors or trace of R*R.T
             score_consistency = np.trace(R @ prev_R.T)
        
        # Weighted combination
        # Prioritize "In Front" and "Consistent"
        if prev_R is not None:
            total_score = score_z + 5.0 * score_consistency + 10.0 * is_in_front
        else:
            total_score = score_z + 10.0 * is_in_front

        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return Rs[best_idx], Ts[best_idx]

class PoseFilter:
    def __init__(self, alpha=0.1):
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
