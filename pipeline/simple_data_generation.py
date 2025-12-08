"""
Usage: 
python ./pipeline/simple_data_generation.py <num_views>
"""

import os
import sys
# Force headless rendering for machines without a display
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image


def get_camera_pose(radius, azimuth, elevation, look_at=np.array([0, 0, 0])):
    """
    Calculates Camera-to-World matrix for a camera looking at a target point.
    radius: distance from look_at point
    azimuth/elevation: in radians
    look_at: 3D point the camera focuses on
    """
    # Convert spherical to cartesian - FIXED elevation formula
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation)  # Y is up
    z = radius * np.cos(elevation) * np.sin(azimuth)
    
    # Position camera relative to look_at point
    camera_pos = look_at + np.array([x, y, z])
    
    # Forward vector (Camera points to look_at)
    fwd = (look_at - camera_pos)
    fwd = fwd / np.linalg.norm(fwd)
    
    # Right/Up vectors (Gram-Schmidt)
    # Assume world UP is (0, 1, 0)
    world_up = np.array([0, 1, 0])
    right = np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right)
    
    # Handle edge case when camera looks straight up/down
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right /= right_norm
    
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    
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


def load_and_fix_mesh(path, target_scale=None):
    """
    Loads a mesh/scene, collapses scenes to a single mesh, 
    and optionally normalizes the scale.
    """
    print(f"Loading {path}...")
    mesh_data = trimesh.load(path, force='mesh', process=True)  # CHANGED: process=True
    
    # FIX: If it's a Scene (multiple objects), merge them into one Mesh
    if isinstance(mesh_data, trimesh.Scene):
        print(f"  > Detected Scene (walls, floor, etc.). Merging into single mesh.")
        # We dump the scene into a single mesh object
        mesh = trimesh.util.concatenate(tuple(mesh_data.geometry.values()))
    else:
        mesh = mesh_data

    # CRITICAL FIX: Clean the mesh to remove artifacts
    print(f"  > Cleaning mesh...")
    mesh.merge_vertices()  # Merge duplicate vertices
    mesh.remove_unreferenced_vertices()  # Remove duplicate faces
    # mesh.remove_degenerate_faces()  # Remove zero-area faces
    
    # IMPORTANT: Ensure vertex colors are preserved or set to white
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        # Set to white instead of gray for better visibility
        mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4)) * [255, 255, 255, 255]

    # FIX: Normalize Scale
    if target_scale:
        extents = mesh.bounds[1] - mesh.bounds[0]
        max_extent = np.max(extents)
        scale_factor = target_scale / max_extent
        print(f"  > Rescaling by {scale_factor:.4f} to fit size {target_scale}")
        
        # Scale
        transform = trimesh.transformations.scale_matrix(scale_factor)
        mesh.apply_transform(transform)
        
        # Center at (0,0,0)
        center = mesh.centroid
        mesh.apply_translation(-center)
        
    return mesh


def create_dataset(
        num_views: int = 10, 
        save_imgs: bool = True, 
        save_dir: str = 'train_views'
    ):
    """Main generation loop."""
    if save_imgs and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # FIXED: Brighter ambient light
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    
    # # --- 1. Load & Scale Cornell Box ---
    # try:
    #     box_trimesh = load_and_fix_mesh('./pipeline/objs/CornellBox/CornellBox-Empty-RG.obj', target_scale=4.0)
        
    #     # FIXED: Use smooth shading and ensure material is loaded
    #     box_mesh = pyrender.Mesh.from_trimesh(box_trimesh, smooth=False)
    #     scene.add(box_mesh, pose=np.eye(4))
    # except Exception as e:
    #     print(f"CRITICAL ERROR loading box: {e}")
    #     return

    # --- 2. Load & Scale Bunny ---
    bunny_trimesh = load_and_fix_mesh('./pipeline/objs/bunny.obj', target_scale=1.0)
    
    # Create red material with slightly different settings
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.2, 0.0, 1.0],  # Red color in RGBA (0-1 range)
        metallicFactor=0.0,
        roughnessFactor=0.1,  # Less rough for smoother appearance
        alphaMode='OPAQUE'  # Ensure no transparency issues
    )
    
    # FIXED: Calculate bunny position more carefully
    bunny_y_offset = -1.2
    bunny_position = np.array([0.0, bunny_y_offset, 0.0])
    
    pose_bunny = np.eye(4)
    pose_bunny[:3, 3] = bunny_position
    
    # Use the material when creating the mesh
    bunny_mesh = pyrender.Mesh.from_trimesh(bunny_trimesh, smooth=True, material=material)
    scene.add(bunny_mesh, pose=pose_bunny)
    
    print(f"Bunny positioned at: {bunny_position}")
    
    # --- 3. FIXED: Reduce light intensity to avoid oversaturation ---
    # Main ceiling light above bunny
    light1 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=50.0)  # Reduced from 200
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = [0.0, 1.5, 0.0]
    scene.add(light1, pose=light_pose1)
    
    # Additional fill lights around bunny
    light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=30.0)  # Reduced from 100
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [1.5, bunny_y_offset, 1.5]
    scene.add(light2, pose=light_pose2)
    
    light3 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=30.0)  # Reduced from 100
    light_pose3 = np.eye(4)
    light_pose3[:3, 3] = [-1.5, bunny_y_offset, -1.5]
    scene.add(light3, pose=light_pose3)

    # --- 4. Camera ---
    H, W = 128, 128
    fx = 100.0
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]])
    camera = pyrender.IntrinsicsCamera(fx, fx, W/2, H/2)
    
    all_rgb, all_rays_o, all_rays_d = [], [], []
    
    print(f"Generating data... Saving images to ./{save_dir}/")
    renderer = pyrender.OffscreenRenderer(W, H)
    
    for i in range(num_views):
        # FIXED: Camera looks at bunny position, stays inside box
        # Box extends roughly from -2 to +2 in each dimension
        radius = np.random.uniform(1.2, 1.8)  # Stay inside box (radius < 2)
        azimuth = np.random.uniform(0, 2*np.pi)
        # Elevation: look from slightly below to above bunny
        elevation = np.random.uniform(-0.2, 0.5)
        
        # CRITICAL FIX: Camera looks at bunny, not origin
        pose = get_camera_pose(
            radius=radius, 
            azimuth=azimuth, 
            elevation=elevation,
            look_at=bunny_position  # Look at bunny!
        )
        
        # Render
        cam_node = scene.add(camera, pose=pose)
        
        color, _ = renderer.render(scene)
        scene.remove_node(cam_node)
        
        # Save Image
        img = Image.fromarray(color)
        img.save(os.path.join(save_dir, f'view_{i:03d}.png'))
        
        # Collect Data
        rays_o, rays_d = generate_rays(H, W, K, pose)
        all_rgb.append(color)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        
        cam_pos = pose[:3, 3]
        print(f"  Rendered view {i+1}/{num_views} | Camera at {cam_pos}")

    renderer.delete()
    
    # Save Arrays
    np.savez('experiment_data.npz', 
             rgb=np.stack(all_rgb), 
             rays_o=np.stack(all_rays_o), 
             rays_d=np.stack(all_rays_d))
    print("Done! Data saved to experiment_data.npz")

if __name__ == "__main__":
    num_views = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    create_dataset(num_views=num_views)