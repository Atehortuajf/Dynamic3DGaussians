import torch
import os
import open3d as o3d
import numpy as np
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
    union_mask = torch.zeros_like(masks[0]).bool()
    for mask, _ in masks:
        union_mask = torch.logical_or(union_mask, mask)
    return union_mask

def get_sparse_depth_mean(pts, cam, mask):
    dists = torch.tensor([]).to(pts.device)
    visible_pts = pts[cam.markVisible(pts)]
    cam_pos = cam.viewmatrix[:, :3, 3]
    projections = project(visible_pts, cam)
    for i in range(len(projections)):
        if mask[projections[i][0], projections[i][1]]:
            dist = (visible_pts[i] - cam_pos).norm()
            dists = torch.cat(dists, dist.unsqueeze(0))
    return dists.mean()

def project(pts, cam):
    # Ensure points are in homogeneous coordinates
    if pts.shape[-1] == 3:
        ones = torch.ones_like(pts[..., :1])
        pts = torch.cat((pts, ones), dim=-1)

    # Multiply points by the projection matrix
    projected_points = torch.matmul(pts, cam.projmatrix.transpose(1, 2))

    # Divide by the w component
    w = projected_points[..., 3:4]
    projected_points = projected_points[..., :3] / w

    # Regularize for cam resolution
    projected_points[..., 0] = int(projected_points[..., 0] * cam.image_width)
    projected_points[..., 1] = int(projected_points[..., 1] * cam.image_height)

    return projected_points[..., :2]

#TODO: Implement these functions
def unproject(points, depths, cam):
    return None

def sample_pts_uniformly(cam, image, num_samples):
    return None

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
