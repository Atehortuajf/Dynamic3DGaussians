import torch
from omegaconf import DictConfig

from helpers import l1_loss_v1
from helpers import weighted_l2_loss_v1
from helpers import weighted_l2_loss_v2
from helpers import l1_loss_v2
from helpers import quat_mult
from helpers import params2rendervar
from helpers import build_rotation

from external import calc_ssim

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def apply_camera_parameters(image: torch.Tensor, params: dict, curr_data: dict) -> torch.Tensor:
    curr_id = curr_data['id']
    return torch.exp(params['cam_m'][curr_id])[:, None, None] * image + params['cam_c'][curr_id][:, None, None]

def compute_loss(cfg:DictConfig, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = l1_loss_v1(rendered, target)
    ssim = 1.0 - calc_ssim(rendered, target)
    return cfg.loss.l1 * l1 + cfg.loss.ssim * ssim

def compute_rigid_loss(fg_pts, rot, variables):
    neighbor_pts = fg_pts[variables["neighbor_indices"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
    return weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"], variables["neighbor_weight"])

def compute_rot_loss(rel_rot, variables):
    return weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None], variables["neighbor_weight"])

def compute_iso_loss(fg_pts, variables):
    neighbor_pts = fg_pts[variables["neighbor_indices"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
    return weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

def compute_floor_loss(fg_pts):
    return torch.clamp(fg_pts[:, 1], min=0).mean()

def compute_bg_loss(bg_pts, bg_rot, variables):
    return l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

def compute_pose_loss(prev_pos, pos, rot):
    return l1_loss_v2(prev_pos[..., :3, 3], pos) + l1_loss_v2(prev_pos[..., :3, :3], build_rotation(rot))


def get_loss(cfg:DictConfig ,params:dict, curr_data:dict, variables:dict, is_initial_timestep:bool):

    losses = {}

    # Pose update
    curr_data['cam'].viewmatrix[..., :3, :3] = build_rotation(params['cam_rot'][curr_data['id']].detach()).T
    curr_data['cam'].viewmatrix[..., 3, :3] = params['cam_pos'][curr_data['id']].detach()
    curr_data['cam'].projmatrix[...] = curr_data['cam'].viewmatrix.bmm(curr_data['proj'])

    losses['pose_cons'] = compute_pose_loss(variables["prev_cam_pos"], params['cam_pos'], params['cam_rot']) * cfg.loss.pose_cons

    # Image
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    image, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    image = apply_camera_parameters(image, params, curr_data)
    
    # Segmentation
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    losses['im'] = compute_loss(cfg, image, curr_data['im']) * cfg.loss.im
    losses['seg'] = compute_loss(cfg, seg, curr_data['seg']) * cfg.loss.seg
    
    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)

        losses['rigid'] = compute_rigid_loss(fg_pts, rot, variables) * cfg.loss.rigid
        losses['rot'] = compute_rot_loss(rel_rot, variables) * cfg.loss.rot
        losses['iso'] = compute_iso_loss(fg_pts, variables) * cfg.loss.iso
        losses['floor'] = compute_floor_loss(fg_pts) * cfg.loss.floor
        
        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = compute_bg_loss(bg_pts, bg_rot, variables) * cfg.loss.bg
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"]) * cfg.loss.soft_col_cons

    loss = sum([v for _, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables