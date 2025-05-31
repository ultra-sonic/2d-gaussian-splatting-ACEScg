#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import colour
from colour import RGB_COLOURSPACES # More specific import
import torch # Added torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    # Field 'image: Image.Image' is REMOVED.
    # Field 'original_image: torch.Tensor' stores the ACEScg tensor on CPU.
    original_image: torch.Tensor
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

# Add input_color_space to function arguments in a later step when call sites are updated
# For now, assume it's available in the local scope. Will be fixed in subsequent steps.
# ^^^ This comment is now being addressed.

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

# output_color_space param removed, hardcoded to "ACEScg"
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, input_color_space="sRGB"):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        pil_image = Image.open(image_path)

        # Convert PIL image to RGB if it's grayscale (L) or grayscale with alpha (LA)
        if pil_image.mode == 'L' or pil_image.mode == 'LA':
            pil_image = pil_image.convert('RGB')

        img_np_input_space = np.array(pil_image, dtype=np.float32) / 255.0

        # Handle cases where image might still have an alpha channel after conversion (e.g. from LA)
        # or if the original was RGBA
        if img_np_input_space.ndim == 3 and img_np_input_space.shape[-1] == 4:
            img_np_input_space = img_np_input_space[..., :3] # Convert to RGB

        # The old 'ndim == 2' check and manual stacking are now replaced by PIL's .convert('RGB')

        # Ensure input_color_space string is a key in RGB_COLOURSPACES
        # For simplicity, assuming valid keys based on previous setup.
        # If input_color_space is 'sRGB', it implies standard sRGB gamma.
        # If 'ACEScg', it implies ACEScg.
        # The `gt_image_tensor` should be in the *linear* version of the input space if applicable,
        # or the direct space if it's already linear (like ACEScg).
        # However, the prompt for previous subtask (loss calculation) stated gt_image is Linear sRGB.
        # Convert img_np_input_space (NumPy, 0-1, input_color_space) to "ACEScg"
        img_np_acescg = colour.RGB_to_RGB(img_np_input_space,
                                          RGB_COLOURSPACES[input_color_space],
                                          RGB_COLOURSPACES["ACEScg"]) # Hardcoded target
        img_np_acescg = np.clip(img_np_acescg, 0.0, 1.0) # Ensure clipping after conversion

        # original_image_tensor_acescg = torch.from_numpy(img_np_acescg).permute(2, 0, 1).contiguous() # This line was previously removed, now re-adding and assigning
        original_image_tensor_acescg = torch.from_numpy(img_np_acescg.copy()).permute(2, 0, 1).contiguous()


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                              original_image=original_image_tensor_acescg, # Store the tensor
                              image_path=image_path, image_name=image_name,
                              width=width, height=height) # width/height from intrinsics
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

# output_color_space param removed, sh_linear_cs will be hardcoded to "ACEScg"
def fetchPly(path, input_color_space="sRGB"):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # Colors from PLY are in input_color_space (e.g. sRGB vertex colors)
    colors_ply_input_space = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    # The internal linear space for SH is now "ACEScg"
    sh_linear_cs = "ACEScg"

    # Convert PLY vertex colors from their input_color_space to "ACEScg"
    colors_for_sh_initialization = colour.RGB_to_RGB(colors_ply_input_space,
                                                     RGB_COLOURSPACES[input_color_space],
                                                     RGB_COLOURSPACES[sh_linear_cs]) # Target is sh_linear_cs ("ACEScg")
    colors_for_sh_initialization = np.clip(colors_for_sh_initialization, 0.0, 1.0)

    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # BasicPointCloud will store colors in "ACEScg"
    return BasicPointCloud(points=positions, colors=colors_for_sh_initialization, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# output_color_space param removed from signature and calls
def readColmapSceneInfo(path, images, eval, llffhold=8, input_color_space="sRGB"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    # output_color_space removed from call to readColmapCameras
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), input_color_space=input_color_space)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        # output_color_space removed from call to fetchPly
        pcd = fetchPly(ply_path, input_color_space)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# output_color_space param removed, hardcoded to "ACEScg"
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", input_color_space="sRGB"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

            # arr is img_np_input_space (H,W,C float32 in [0,1]) and is in input_color_space
            img_np_input_space = arr

            # Convert img_np_input_space (NumPy, 0-1, input_color_space) to "ACEScg"
            img_np_acescg = colour.RGB_to_RGB(img_np_input_space,
                                              RGB_COLOURSPACES[input_color_space],
                                              RGB_COLOURSPACES["ACEScg"]) # Hardcoded target

            img_np_acescg = colour.RGB_to_RGB(img_np_input_space,
                                              RGB_COLOURSPACES[input_color_space],
                                              RGB_COLOURSPACES["ACEScg"]) # Hardcoded target
            img_np_acescg = np.clip(img_np_acescg, 0.0, 1.0) # Ensure clipping

            # img_pil_acescg = Image.fromarray((np.clip(img_np_acescg, 0, 1) * 255.0).astype(np.uint8)) # No longer needed
            original_image_tensor_acescg = torch.from_numpy(img_np_acescg.copy()).permute(2, 0, 1).contiguous()

            # Use initial image 'image' (PIL) for size, as img_pil_acescg is no longer created
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        original_image=original_image_tensor_acescg, # Store the tensor
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1])) # width/height from initial PIL image
            
    return cam_infos

# output_color_space param removed from signature and calls
def readNerfSyntheticInfo(path, white_background, eval, extension=".png", input_color_space="sRGB"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, input_color_space)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, input_color_space)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        # output_color_space removed from call to fetchPly
        pcd = fetchPly(ply_path, input_color_space)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}