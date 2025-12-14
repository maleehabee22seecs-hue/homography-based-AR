import numpy as np
import open3d as o3d
import cv2
import copy

class MeshRenderer3D:
    def __init__(self, mesh_path, K, width, height):
        print(f"[MeshRenderer3D] Loading mesh: {mesh_path}")

        # 1. Try newer Open3D Tensor API (supports quads)
        try:
            mesh_t = o3d.t.io.read_triangle_mesh(mesh_path)
            mesh = mesh_t.to_legacy()
            print("[MeshRenderer3D] Loaded mesh via Tensor API (quad support).")
        except Exception:
            print("[MeshRenderer3D] Tensor API failed, falling back to legacy loader.")
            mesh = o3d.io.read_triangle_mesh(mesh_path)

        # 2. Force triangulation
        try:
            mesh.compute_vertex_normals()
        except:
            pass

        if mesh.is_empty():
            raise RuntimeError("MeshRenderer3D: mesh is EMPTY. OBJ likely invalid.")

        # 3. Store base mesh (normalized)
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        
        # Center mesh at origin
        center = self.mesh.get_center()
        self.mesh.translate(-center)

        # Normalize mesh size (Fix for "huge object covering screen")
        min_bound = self.mesh.get_min_bound()
        max_bound = self.mesh.get_max_bound()
        dims = max_bound - min_bound
        max_dim = np.max(dims)
        if max_dim > 1e-6:
            scale_factor = 1.0 / max_dim
            self.mesh.scale(scale_factor, center=(0,0,0))
            print(f"[MeshRenderer3D] Normalized mesh. Scale factor: {scale_factor:.4f}")
        
        # Keep a copy of vertices to avoid drift
        self.base_vertices = np.asarray(self.mesh.vertices).copy()

        self.width = int(width)
        self.height = int(height)
        self.K = K.copy().astype(np.float64)

        # Legacy Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        
        # Position offset (where the object sits in 3D world, relative to Ref Frame at 0,0,0)
        self.object_pos = np.array([0.0, 0.0, 0.0]) # Default origin (bad, will fix with set_anchor_pos)
        
        # Add the geometry ONCE -- Paint it if it has no colors
        if not self.mesh.has_vertex_colors() and not self.mesh.has_textures():
             print("[MeshRenderer3D] Mesh has no colors/textures. Painting it cyan.")
             self.mesh.paint_uniform_color([0.0, 1.0, 1.0]) # Cyan
        
        self.vis.add_geometry(self.mesh)
        
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0]) 
        opt.light_on = True

        print("[MeshRenderer3D] Mesh ready.")

    def render(self, R, t, rot_x=0.0, rot_y=0.0, rot_z=0.0, scale=1.0):
        # 1. Calculate Transform
        # R_user = Rz * Ry * Rx
        Rx = self.mesh.get_rotation_matrix_from_xyz((np.deg2rad(rot_x), 0, 0))
        Ry = self.mesh.get_rotation_matrix_from_xyz((0, np.deg2rad(rot_y), 0))
        Rz = self.mesh.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(rot_z)))
        
        R_user = Rz @ Ry @ Rx
        
        # Apply "Upside Down" fix (180 around X)
        R_fix = self.mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        R_obj = R_fix @ R_user
        
        # We need to transform the BASE vertices to the new position
        # V_new = (R_obj * V_base) * scale
        
        # Apply strict transform to vertices
        # Note: Open3D arrays are (N,3), so V * R_obj.T
        verts = self.base_vertices @ R_obj.T * scale
        
        # ADDED: Translate object to its anchored position in the World (Reference) frame
        verts += self.object_pos
        
        # Update mesh in place
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh.compute_vertex_normals() # recompute normals for lighting
        self.vis.update_geometry(self.mesh)

        # 2. Setup Camera
        ctr = self.vis.get_view_control()
        
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, 
            self.K[0,0], self.K[1,1], 
            self.K[0,2], self.K[1,2]
        )
        
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = R
        extrinsic[:3,3] = t.flatten()
        param.extrinsic = extrinsic
        
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        
        # 3. Render
        self.vis.poll_events()
        self.vis.update_renderer()
        
        img = self.vis.capture_screen_float_buffer(do_render=True)
        rgb = (np.asarray(img) * 255).astype(np.uint8)

        depth_img = self.vis.capture_depth_float_buffer(do_render=False)
        depth = np.asarray(depth_img)
        
        mask = depth > 0
        alpha = np.zeros((self.height, self.width), dtype=np.uint8)
        alpha[mask] = 255
        
        rgba_out = np.dstack((rgb, alpha))
        
        return rgba_out
    
    def set_anchor_pos(self, pos):
        """
        Set the static 3D position of the object in the reference frame.
        pos: (3,) numpy array or list [x, y, z]
        """
        self.object_pos = np.array(pos, dtype=np.float64)
        print(f"[MeshRenderer3D] Object anchor set to: {self.object_pos}")

    def close(self):
        self.vis.destroy_window()
