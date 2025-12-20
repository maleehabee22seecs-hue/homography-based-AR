import cv2
import numpy as np
import argparse
import os
import sys
from placement import PlacementController
from tracker import FeatureTracker
from renderer_3d import MeshRenderer3D
from pose import decompose_homography, PoseFilter

# Default Paths (Relative to the script execution or hardcoded fallback)
DEFAULT_VIDEO = os.path.join(os.path.dirname(__file__), "../data/input_video.mp4")
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "../data/chair.glb")

# Camera Calibration Data (Samsung A52 / Example)
# NOTE: In a real production app, this should be loaded from a file.
K_cal = np.array([[969.8165, 0.0, 292.3732],
                  [  0.0, 983.1865, 675.59],
                  [  0.0, 0.0, 1.0 ]])

dist_cal = np.array([ 0.2558, -3.947,  -0.0013,  0.0069, 14.6611])
w_cal, h_cal = 564, 1280

def get_camera_matrix(w_vid, h_vid):
    scale_x = w_vid / w_cal
    scale_y = h_vid / h_cal
    K = K_cal.copy()
    K[0,0] *= scale_x; K[0,1] *= scale_x
    K[1,1] *= scale_y; K[1,2] *= scale_y
    return K

def get_generic_calibration(w_vid, h_vid, fov_deg=60.0):
    f = max(w_vid, h_vid) / (2 * np.tan(np.radians(fov_deg/2)))
    cx = w_vid / 2.0
    cy = h_vid / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1.0]])
    return K

def main():
    parser = argparse.ArgumentParser(description="Homography-based AR Object Placement")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="Path to input video")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to 3D OBJ model")
    args = parser.parse_args()

    # Verify files
    # Try to parse video as integer for webcam
    video_source = args.video
    if args.video.isdigit():
        video_source = int(args.video)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
        
    h_vid, w_vid = frame0.shape[:2]
    print(f"Video opened. Size: {w_vid}x{h_vid}")
    
    # Setup Camera Matrix
    # Setup Camera Matrix
    if isinstance(video_source, int):
        print("Using Generic Webcam Calibration...")
        K = get_generic_calibration(w_vid, h_vid)
        # Assume no distortion for webcam if unknown
        dist_cal_run = np.zeros(5) 
    else:
        print("Using Default Calibration...")
        K = get_camera_matrix(w_vid, h_vid)
        dist_cal_run = dist_cal

    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist_cal_run, (w_vid,h_vid), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_cal_run, None, newK, (w_vid,h_vid), cv2.CV_16SC2)

    # Initialize Modules
    placement = PlacementController(frame0)
    tracker = FeatureTracker()
    try:
        renderer = MeshRenderer3D(args.model, K, w_vid, h_vid)
    except Exception as e:
        print(f"Failed to initialize renderer: {e}")
        return
    
    # Interaction State
    obj_state = {
        "scale": 10.0,
        "rx": 0.0, "ry": 0.0, "rz": 0.0
    }

    # Stabilization & Geometry
    pose_filter = PoseFilter(alpha=0.7) 
    # FIXED_DEPTH = 20.0 # Virtual distance to place object
    
    cv2.namedWindow("AR3D")
    cv2.setMouseCallback("AR3D", placement.mouse_callback)

    frame_idx = 0
    print("\nControls:")
    print("  [Click]: Place & Lock Object here")
    # print("  [C]: Lock/Confirm Placement") # Removed as it's now click-to-lock
    print("  [R]: Reset")
    print("  [+/-]: Scale")
    print("  [W/S]: Rotate X (Pitch)")
    print("  [A/D]: Rotate Y (Yaw)")
    print("  [G/E]: Rotate Z (Roll)")
    print("  [ESC]: Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Video Loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if placement.locked:
                 tracker.initialized = False
            continue
            
        frame_idx += 1
        
        # Undistort
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        display = frame.copy()

        # Logic
        if placement.locked:
            # Object IS locked
            if not tracker.initialized:
                tracker.initialize(placement.initial_frame, placement.roi_corners)
                renderer.set_anchor_pos([0,0,0])
                pose_filter.reset()

            # Tracking
            H, p_init, p_curr = tracker.track(frame)
            
            if H is not None:
                # 1. Map anchor to current frame
                ux, uy = placement.anchor
                anchor_hom = np.array([ux, uy, 1.0])
                curr_anchor_hom = H @ anchor_hom
                cur_u, cur_v = curr_anchor_hom[:2] / curr_anchor_hom[2]
                
                # 2. Rotation & Translation
                prev_R = pose_filter.R_curr if pose_filter.initialized else None
                R, _ = decompose_homography(H, K, prev_R)
                
                # 3. Forced Translation (Ray casting)
                # We force the object to lie on the ray passing through (cur_u, cur_v) at fixed depth
                FIXED_DEPTH = 20.0 
                K_inv = np.linalg.inv(K)
                ray = K_inv @ np.array([cur_u, cur_v, 1.0])
                t_forced = ray * FIXED_DEPTH
                
                # 4. Filter
                R_smooth, t_smooth = pose_filter.update(R, t_forced)
                
                # 5. Render
                try:
                    model_rgba = renderer.render(R_smooth, t_smooth, 
                                               rot_x=obj_state["rx"], 
                                               rot_y=obj_state["ry"], 
                                               rot_z=obj_state["rz"], 
                                               scale=obj_state["scale"])

                    mask = model_rgba[:,:,3] > 0
                    display[mask] = model_rgba[:,:,:3][mask]
                except Exception as e:
                    pass # Prevent crash on render fail
                    
            else:
                 cv2.putText(display, "Tracking Lost", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        else:
            # Object NOT locked (Preview Mode)
            if placement.anchor:
                 display = placement.draw_preview(display)

        # UI Overlay
        cv2.putText(display, "FPS: N/A", (w_vid-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        status_text = "LOCKED" if placement.locked else "SELECT POINT"
        cv2.putText(display, f"Status: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Debug State
        state_str = f"S:{obj_state['scale']:.1f} RX:{obj_state['rx']:.1f} RY:{obj_state['ry']:.1f} RZ:{obj_state['rz']:.1f}"
        cv2.putText(display, state_str, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)

        # Controls Overlay
        controls = [
            "Controls:",
            "[Click]: Place Object",
            "[R]: Reset",
            "[+/-]: Scale",
            "[W/S]: Rotate X",
            "[A/D]: Rotate Y",
            "[G/E]: Rotate Z",
            "[ESC]: Quit"
        ]
        
        y_start = h_vid - (len(controls) * 25) - 20 # Position at bottom left
        for i, text in enumerate(controls):
            cv2.putText(display, text, (20, y_start + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # Show Controls overlay
        cv2.imshow("AR3D", display)
        
        key = cv2.waitKey(10)
        if key == 27: # ESC
            break
        elif key == ord('c'):
            placement.lock()
        elif key == ord('r'):
            print("Resetting...")
            placement.locked = False
            placement.anchor = None
            tracker.initialized = False
            pose_filter.reset()
        
        # Transformations
        elif key == ord('='): obj_state["scale"] += 0.5
        elif key == ord('-'): obj_state["scale"] = max(0.1, obj_state["scale"] - 0.5)
        elif key == ord('w'): obj_state["rx"] += 10.0
        elif key == ord('s'): obj_state["rx"] -= 10.0
        elif key == ord('a'): obj_state["ry"] -= 10.0
        elif key == ord('d'): obj_state["ry"] += 10.0
        elif key == ord('g'): obj_state["rz"] -= 10.0
        elif key == ord('e'): obj_state["rz"] += 10.0

    cap.release()
    renderer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

