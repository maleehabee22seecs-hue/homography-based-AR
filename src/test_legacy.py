
import open3d as o3d
import numpy as np

try:
    print("Creating legacy Visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    print("Window created.")
    
    mesh = o3d.geometry.TriangleMesh.create_box()
    vis.add_geometry(mesh)
    print("Geometry added.")
    
    vis.poll_events()
    vis.update_renderer()
    print("Update renderer done.")
    
    img = vis.capture_screen_float_buffer(do_render=True)
    print(f"Captured image shape: {np.asarray(img).shape}")
    
    vis.destroy_window()
    print("Success!")
except Exception as e:
    print(f"Legacy visualizer failed: {e}")
