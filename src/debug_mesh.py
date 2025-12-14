
import open3d as o3d

mesh_path = "C:/Users/Maleeha/OneDrive/Desktop/A/data/object3d.obj"

try:
    print("Loading with Tensor API...")
    mesh_t = o3d.t.io.read_triangle_mesh(mesh_path)
    print(f"Tensor mesh loaded. Vertices: {len(mesh_t.vertex.positions)}")
    
    print("Converting to legacy...")
    mesh_legacy = mesh_t.to_legacy()
    print(f"Legacy mesh from tensor. Is empty? {mesh_legacy.is_empty()}")
    if not mesh_legacy.is_empty():
        print(f"Legacy Vertices: {len(mesh_legacy.vertices)}")
        print("Success!")
except Exception as e:
    print(f"Conversion failed: {e}")
