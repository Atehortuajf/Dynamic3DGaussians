import torch
import os
import open3d as o3d
import numpy as np
import cv2

import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def setup_camera(
        width:int,
        height:int,
        instrinsics:np.ndarray,
        world_2_cam:np.ndarray,
        near:float, # Near and far clipping planes for depth in the camera's view frustum. (in meters?)
        far:float      
    ) -> Camera:
    # Focal length, (x, y in pixels) --- optical center (x, y)
    fx, fy, cx, cy = instrinsics[0][0], instrinsics[1][1], instrinsics[0][2], instrinsics[1][2]

    world_2_cam_tensor:torch.Tensor = torch.tensor(world_2_cam).cuda().float()
    
    # position of the camera center in the world coordinates.
    cam_center = torch.inverse(world_2_cam_tensor)[:3, 3]
    world_2_cam_tensor = world_2_cam_tensor.unsqueeze(0).transpose(1, 2)

    # This matrix is used to map 3D world coordinates to 2D camera coordinates, factoring in the depth.
    opengl_proj = torch.tensor([
        [ 2 * fx / width,              0.0,              -(width - 2 * cx) / width,   0.0 ],
        [ 0.0,                         2 * fy / height,  -(height - 2 * cy) / height, 0.0 ],
        [ 0.0,                         0.0,              far / (far - near),         -(far * near) / (far - near) ],
        [ 0.0,                         0.0,              1.0,                         0.0 ]
    ]).cuda().float().unsqueeze(0).transpose(1, 2)

    # This will give a matrix that transforms from world coordinates
    # directly to normalized device coordinates (NDC)
    full_proj = world_2_cam_tensor.bmm(opengl_proj)
    
    cam = Camera(
        image_height=height,
        image_width=width,
        tanfovx=width / (2 * fx),
        tanfovy=height / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_2_cam_tensor,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def params2rendervar(params:dict) -> dict:
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def union_masks(masks):
    union_mask = torch.zeros_like(masks[0][0][0]).bool()
    for mask, _ in masks:
        union_mask = torch.logical_or(union_mask, mask[0])
    return union_mask

def homogenize_points(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def unproject(coordinates, depth, intrinsics):
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = torch.einsum(
        "... i j, ... j -> ... i", intrinsics.inverse(), coordinates
    )

    # Apply the supplied depth values.
    return ray_directions * depth[..., None]

#TODO: Sample from mask instead of the whole image (how???)
def sample_pts_uniformly(entry, depth, mask, num_samples):
    xy = torch.rand((num_samples, 2), dtype=torch.float32, device="cuda")
    mask_cond = (mask[(xy[..., 0] * mask.shape[0]).int(), (xy[..., 1] * mask.shape[1]).int()])
    # Expected value of xy_valid should match paper
    xy_valid = xy[mask_cond] * 2 - 1 #[-1,1] for grid_sample
    # Sample from depth map
    sampled_depth = F.grid_sample(depth[None, None], xy_valid[None, None], align_corners=True).squeeze()
    # Sample color from image
    sampled_color = F.grid_sample(entry['im'][None], xy_valid[None, None], align_corners=True).squeeze()
    # Unproject to 3D
    sampled_points_cam = unproject(xy_valid, sampled_depth, entry['intrinsics'])
    sampled_points = torch.einsum('...ij,...kj->...ki', entry['w2c'].inverse(),
                                  homogenize_points(sampled_points_cam))[..., :3]

    return sampled_points, sampled_color, torch.ones_like(sampled_depth)

def record_video(vid_frames, seq, exp):
    video_path = f"./output/{exp}/{seq}/video.avi"
    frame_size = (vid_frames[0].shape[1], vid_frames[0].shape[2])
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, frame_size)
    for frame in vid_frames:
        frame_write = frame.detach().cpu().numpy().transpose(1, 2, 0)
        video_writer.write((255. * frame_write).astype(np.uint8))
    video_writer.release()

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def build_rotation(quaternions):
    """
    Convert a batch of quaternions to rotation matrices.
    
    Args:
        quaternions (torch.Tensor): A tensor of shape (..., 4) containing the quaternions.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) containing the rotation matrices.
    """
    batch_dim = quaternions.shape[:-1]
    q = quaternions.reshape(-1, 4)
    
    # Normalize the quaternions to ensure they are unit quaternions
    q = q / q.norm(dim=1, keepdim=True)
    
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Calculate the elements of the rotation matrix
    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    
    return R.reshape(*batch_dim, 3, 3)

def build_quaternion(rot_mats):
    """
    Convert a batch of rotation matrices to quaternions.
    
    Args:
        rot_mats (torch.Tensor): A tensor of shape (..., 3, 3) containing the rotation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4) containing the quaternions.
    """
    batch_dim = rot_mats.shape[:-2]
    m = rot_mats.reshape(-1, 3, 3)
    
    # The trace of the matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    s = torch.zeros_like(trace)
    quaternion = torch.zeros((m.shape[0], 4), dtype=m.dtype, device=m.device)
    
    # For trace > 0
    mask = trace > 0
    s[mask] = torch.sqrt(trace[mask] + 1.0) * 2  # s=4*qw
    quaternion[mask, 0] = 0.25 * s[mask]
    quaternion[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quaternion[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quaternion[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]
    
    # For trace <= 0
    mask = ~mask
    mask_0 = (m[mask, 0, 0] > m[mask, 1, 1]) & (m[mask, 0, 0] > m[mask, 2, 2])
    mask_1 = ~mask_0 & (m[mask, 1, 1] > m[mask, 2, 2])
    mask_2 = ~mask_0 & ~mask_1
    
    s[mask][mask_0] = torch.sqrt(1.0 + m[mask][mask_0, 0, 0] - m[mask][mask_0, 1, 1] - m[mask][mask_0, 2, 2]) * 2  # s=4*qx
    quaternion[mask][mask_0, 0] = (m[mask][mask_0, 2, 1] - m[mask][mask_0, 1, 2]) / s[mask][mask_0]
    quaternion[mask][mask_0, 1] = 0.25 * s[mask][mask_0]
    quaternion[mask][mask_0, 2] = (m[mask][mask_0, 0, 1] + m[mask][mask_0, 1, 0]) / s[mask][mask_0]
    quaternion[mask][mask_0, 3] = (m[mask][mask_0, 0, 2] + m[mask][mask_0, 2, 0]) / s[mask][mask_0]
    
    s[mask][mask_1] = torch.sqrt(1.0 + m[mask][mask_1, 1, 1] - m[mask][mask_1, 0, 0] - m[mask][mask_1, 2, 2]) * 2  # s=4*qy
    quaternion[mask][mask_1, 0] = (m[mask][mask_1, 0, 2] - m[mask][mask_1, 2, 0]) / s[mask][mask_1]
    quaternion[mask][mask_1, 1] = (m[mask][mask_1, 0, 1] + m[mask][mask_1, 1, 0]) / s[mask][mask_1]
    quaternion[mask][mask_1, 2] = 0.25 * s[mask][mask_1]
    quaternion[mask][mask_1, 3] = (m[mask][mask_1, 1, 2] + m[mask][mask_1, 2, 1]) / s[mask][mask_1]
    
    s[mask][mask_2] = torch.sqrt(1.0 + m[mask][mask_2, 2, 2] - m[mask][mask_2, 0, 0] - m[mask][mask_2, 1, 1]) * 2  # s=4*qz
    quaternion[mask][mask_2, 0] = (m[mask][mask_2, 1, 0] - m[mask][mask_2, 0, 1]) / s[mask][mask_2]
    quaternion[mask][mask_2, 1] = (m[mask][mask_2, 0, 2] + m[mask][mask_2, 2, 0]) / s[mask][mask_2]
    quaternion[mask][mask_2, 2] = (m[mask][mask_2, 1, 2] + m[mask][mask_2, 2, 1]) / s[mask][mask_2]
    quaternion[mask][mask_2, 3] = 0.25 * s[mask][mask_2]
    
    return quaternion.reshape(*batch_dim, 4)


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params(output_params, seq, exp):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params", **to_save)
