import cv2
import numpy as np
from placement import PlacementController
from tracker import FeatureTracker
from renderer_3d import MeshRenderer3D
from pose import decompose_homography, PoseFilter

VIDEO = "C:/Users/Maleeha/OneDrive/Desktop/A/data/input_video.mp4"
MODEL = "C:/Users/Maleeha/OneDrive/Desktop/A/data/object3d.obj"
cap = None # Will be initialized in main
# ret, frame0 = cap.read() # Moved to main
# if not ret:
#     raise RuntimeError("Can't open video")
# h, w = frame0.shape[:2] # This logic is problematic here if video capture isn't open
# Just placeholder needed for width/height? 
# K_cal needs w_vid, h_vid. 
# Let's open briefly to get size then release, or just do it inside main logic?
# The code below uses w_vid, h_vid immediately.
cap_temp = cv2.VideoCapture(VIDEO)
ret, frame0 = cap_temp.read()
if ret:
    h_vid, w_vid = frame0.shape[:2]
    print("Video frame size (w,h):", (w_vid, h_vid))
cap_temp.release()
   



# https://stackoverflow.com/questions/78072261/how-to-find-cameras-intrinsic-matrix-from-focal-length
import numpy as np

# calibrated values (from multi-image calibration)
K_cal = np.array([[969.8165, 0.0, 292.3732],
                  [  0.0, 983.1865, 675.59],
                  [  0.0, 0.0, 1.0 ]])

dist_cal = np.array([ 0.2558, -3.947,  -0.0013,  0.0069, 14.6611])

# ret, frame0 = cap.read() # Redundant, already got frame0 from cap_temp
# h_vid, w_vid = frame0.shape[:2] # Already set above


# if calibration image size differs, scale K:
w_cal, h_cal = 564, 1280   # values used for calibration
scale_x = w_vid / w_cal
scale_y = h_vid / h_cal
K = K_cal.copy()
K[0,0] *= scale_x; K[0,1] *= scale_x
K[1,1] *= scale_y; K[1,2] *= scale_y

# optional undistort maps
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist_cal, (w_vid,h_vid), alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist_cal, None, newK, (w_vid,h_vid), cv2.CV_16SC2)



def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO}")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
        
    print(f"Video opened successfully. Frame shape: {frame0.shape}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")
    h, w = frame0.shape[:2]

    placement = PlacementController(frame0)
    tracker = FeatureTracker()
    renderer = MeshRenderer3D(MODEL, K, w, h)
    
    # Interaction State
    obj_scale = 10.0
    obj_angle_x = 0.0
    obj_angle_y = 0.0
    obj_angle_z = 0.0

    # Stabilization & Geometry
    pose_filter = PoseFilter(alpha=0.7) # Higher alpha = less lag, more responsiveness
    Z_PLANE = 20.0 # Assumed distance to the plane for scale recovery
    
    last_frame = frame0.copy()

    cv2.namedWindow("AR3D")
    cv2.setMouseCallback("AR3D", placement.mouse_callback)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video
            print("End of video, looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Reset tracker on loop, as the scene changes instantly
            if placement.locked:
                 print("Video looped: Resetting tracker.")
                 tracker.initialized = False
                 # Optional: Unset lock if you want user to re-lock
                 # placement.locked = False
                 # placement.anchor = None
            
            continue
            
        frame_idx += 1
        
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        display = frame.copy()

        if placement.anchor and not placement.locked:
            display = placement.draw_preview(display)

        if placement.locked:
            if not tracker.initialized:
                tracker.initialize(placement.initial_frame, placement.roi_corners)
                
                # UNPROJECT anchor to 3D
                # (Logic kept for reference but we lock to 2D ray now)
                
                # We treat the object as centered at the origin of the coordinate system
                # and move it using 't' in the loop.
                renderer.set_anchor_pos([0,0,0])
                pose_filter.reset()

            # Tracker now returns H from Initial -> Current
            H, p_init, p_curr = tracker.track(frame)
            
            if H is not None:
                # 1. Compute robust 2D anchor position using H
                # This guarantees the point stays visually "stuck" to the texture
                ux, uy = placement.anchor
                anchor_hom = np.array([ux, uy, 1.0])
                curr_anchor_hom = H @ anchor_hom
                cur_u, cur_v = curr_anchor_hom[:2] / curr_anchor_hom[2]
                
                # 2. Recover Rotation (Orientation) from Homography
                # We use the previous R to maintain consistency
                prev_R = pose_filter.R_curr if pose_filter.initialized else None
                R, _ = decompose_homography(H, K, prev_R)
                
                # 3. Construct Translation to force object to appear at (cur_u, cur_v) at fixed depth
                # We place the object at (0,0,0) in World, so X_cam = t_final.
                # We want t_final to project to (cur_u, cur_v).
                # t_final = K_inv * [u, v, 1] * Fixed_Depth
                FIXED_DEPTH = 20.0 # Fixed depth to ensure visibility (meters/units)
                K_inv = np.linalg.inv(K)
                ray = K_inv @ np.array([cur_u, cur_v, 1.0])
                
                # ray is (x_norm, y_norm, 1.0). Scaling by Z gives (X, Y, Z).
                t_forced = ray * FIXED_DEPTH
                
                # 4. Filter/Smooth
                # For translation, we use the forced 2D position to prevent drift
                R_smooth, t_smooth = pose_filter.update(R, t_forced)
                
                # Debug Logging
                if frame_idx % 30 == 0:
                    print(f"\n[Debug Frame {frame_idx}]")
                    print(f"  Tracked 2D: ({cur_u:.1f}, {cur_v:.1f})")
                    print(f"  t_forced: {t_smooth.flatten()}")
                    # Reset object anchor to zero since we are moving the camera frame directly to the object center
                    renderer.object_pos = np.array([0.0, 0.0, 0.0])
                
                # Render (Note: we treat object as centered at 0,0,0, so we pass t_smooth as the translation)
                # Render
                print(f"[Debug Rot] X:{obj_angle_x} Y:{obj_angle_y} Z:{obj_angle_z}")
                model_rgba = renderer.render(R_smooth, t_smooth, 
                                           rot_x=obj_angle_x, 
                                           rot_y=obj_angle_y, 
                                           rot_z=obj_angle_z, 
                                           scale=obj_scale)

                mask = model_rgba[:,:,3] > 0
                display[mask] = model_rgba[:,:,:3][mask]

            else:
                 cv2.putText(display, "Tracking Lost", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # GUI Instructions
        # GUI Instructions
        cv2.putText(display, "Press 'c' to lock", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(display, f"Scale (+/-): {obj_scale:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(display, f"Rot X(w/s) Y(a/d) Z(g/e)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(display, f"Angles: {obj_angle_x:.0f}, {obj_angle_y:.0f}, {obj_angle_z:.0f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("AR3D", display)
        last_frame = frame.copy()

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c'):
            placement.lock()
            # Reset tracker if unlocking?
            if not placement.locked:
                tracker.initialized = False
        
        # Interaction Keys
        elif key == ord('='): # Scale Up (+)
            obj_scale += 1.0
        elif key == ord('-'): # Scale Down (-)
            obj_scale = max(1.0, obj_scale - 1.0)
            
        # Rotation Controls
        elif key == ord('w'): # Pitch Up (X+)
            obj_angle_x += 5.0
        elif key == ord('s'): # Pitch Down (X-)
            obj_angle_x -= 5.0
        elif key == ord('a'): # Yaw Left (Y-)
            obj_angle_y -= 5.0
        elif key == ord('d'): # Yaw Right (Y+)
            obj_angle_y += 5.0
        elif key == ord('g'): # Roll Left (Z-)
            obj_angle_z -= 5.0
        elif key == ord('e'): # Roll Right (Z+)
            obj_angle_z += 5.0
            
        elif key == 27: # ESC to quit
            break
            
    print("Loop finished. Releasing resources.")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
