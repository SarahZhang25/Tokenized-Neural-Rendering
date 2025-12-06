import os
# Force headless rendering for machines without a display
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image

def get_camera_pose(radius, azimuth, elevation):
    """
    Calculates Camera-to-World matrix for a camera looking at origin.
    radius: distance from center
    azimuth/elevation: in radians
    """
    # Convert spherical to cartesian
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) # Y is up
    z = radius * np.sin(elevation) * np.sin(azimuth)
    camera_pos = np.array([x, y, z])
    
    # Forward vector (Camera points to origin)
    fwd = -camera_pos / np.linalg.norm(camera_pos)
    
    # Right/Up vectors (Gram-Schmidt)
    # Assume world UP is (0, 1, 0)
    world_up = np.array([0, 1, 0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    
    # 4x4 Matrix
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up # Camera Up
    pose[:3, 2] = -fwd # Camera looks down -Z in standard OpenGL
    pose[:3, 3] = camera_pos
    return pose

def generate_rays(H, W, K, pose):
    """
    Generate rays for every pixel.
    H, W: Height, Width
    K: Intrinsic Matrix (3x3)
    pose: Camera-to-World (4x4)
    """
    # 1. Create Grid of Pixel Coordinates
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), 
                       np.arange(H, dtype=np.float32), 
                       indexing='xy')
    
    # 2. Convert to Camera Coordinates
    # (x - cx) / fx
    dirs = np.stack([(i - K[0, 2]) / K[0, 0], 
                     -(j - K[1, 2]) / K[1, 1], # Flip Y for image coords
                     -np.ones_like(i)], -1)    # Look down -Z
    
    # 3. Rotate rays to World Space
    # Apply rotation part of pose matrix
    rays_d = np.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
    
    # 4. Normalize
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    
    # 5. Ray Origins (Camera Position)
    rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
    
    return rays_o, rays_d

def create_dataset(
        num_views: int = 10, 
        save_imgs: bool = True, 
        save_dir: str = 'train_views'
    ):
    """Main generation loop."""
    if save_imgs and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load Mesh
    mesh_trimesh = trimesh.load('examples/objects/classical/bunny.obj') # REPLACE WITH YOUR OBJ
    
    
    # --- SAFETY FIX 1: Normalize Scale ---
    # Many meshes are in mm (size ~100) or m (size ~0.1). 
    # We force it to fit in a unit box [-0.5, 0.5]
    extents = mesh_trimesh.bounds[1] - mesh_trimesh.bounds[0] # [x_width, y_height, z_depth]
    max_extent = np.max(extents)
    scale_factor = 1.0 / max_extent
    print(f"Original Size: {extents}. Scaling by: {scale_factor}")
    
    # Apply Scale
    mesh_trimesh.apply_transform(trimesh.transformations.scale_matrix(scale_factor))
    
    # Center it at 0,0,0
    center = mesh_trimesh.centroid
    mesh_trimesh.apply_translation(-center)
    
    # --- SAFETY FIX 2: Force Color ---
    # If the OBJ has a black material, it will render black. Force it to Grey.
    if hasattr(mesh_trimesh.visual, 'vertex_colors'):
        # Overwrite with Grey (128, 128, 128) + Alpha (255)
        mesh_trimesh.visual.vertex_colors = np.ones((len(mesh_trimesh.vertices), 4)) * [128, 128, 128, 255]
    
    # --- Setup Scene ---
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0]) # Force BLACK background to see contrast
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene.add(mesh)
    
    # --- SAFETY FIX 3: Directional Light ---
    # Directional lights don't have a position, only a direction.
    # This guarantees light hits the surface regardless of scale.
    # Intensity=5.0 is bright.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=50.0)
    scene.add(light, pose=np.eye(4)) # Default points down -Z (from camera view usually)

    # Camera Intrinsics
    H, W = 128, 128
    fx = 100.0 # Focal length
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]])
    camera = pyrender.IntrinsicsCamera(fx, fx, W/2, H/2)
    
    all_rgb = []
    all_rays_o = []
    all_rays_d = []
    
    # Generate views
    
    for i in range(num_views):
        # Random pose on sphere
        azimuth = np.random.uniform(0, 2*np.pi)
        elevation = np.random.uniform(0.1, np.pi/2) # Upper hemisphere
        pose = get_camera_pose(radius=2.0, azimuth=azimuth, elevation=elevation)
        
        # 1. Render Image (Ground Truth)
        cam_node = scene.add(camera, pose=pose)
        renderer = pyrender.OffscreenRenderer(W, H)
        color, _ = renderer.render(scene)
        scene.remove_node(cam_node) # Clean up for next iter
        
        # 2. Calculate Rays
        rays_o, rays_d = generate_rays(H, W, K, pose)
        
        all_rgb.append(color)      # [H, W, 3]
        all_rays_o.append(rays_o)  # [H, W, 3]
        all_rays_d.append(rays_d)  # [H, W, 3]

        # Save image for inspection
        img = Image.fromarray(color)

        if save_imgs:
            img.save(os.path.join(save_dir, f'view_{i:03d}.png'))

    # Save to disk
    np.savez('experiment_data.npz', 
             rgb=np.stack(all_rgb), 
             rays_o=np.stack(all_rays_o), 
             rays_d=np.stack(all_rays_d))
    print("Data Saved!")

if __name__ == "__main__":
    create_dataset(num_views=1)