import os
import random
import copy

import hydra
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from helpers import setup_camera
from helpers import o3d_knn
from helpers import params2rendervar
from helpers import params2cpu
from helpers import save_params

from external import build_quaternion
from external import build_rotation
from external import calc_psnr
from external import densify
from external import update_params_and_optimizer
from external import lr_scheduler

from priors import get_priors

from loss import get_loss

schedulers = {}

def construct_timestep_dataset(timestep:int, priors:dict, cfg:DictConfig) -> list[dict]:
    dataset_entries = []
    for camera_id in priors['fn'][timestep].keys():
        width, height, intrinsics, extrinsics = priors['w'], priors['h'], priors['k'][timestep][camera_id], priors['w2c'][timestep][camera_id]
        camera = setup_camera(width, height, intrinsics, extrinsics, near=cfg.train.near, far=cfg.train.far)
        opengl_proj = camera.viewmatrix.inverse().bmm(camera.projmatrix)
        
        filename = priors['fn'][timestep][camera_id]
        
        image = np.array(copy.deepcopy(Image.open(f"./data/{cfg.train.seq}/ims/{filename}")))
        image_tensor = torch.tensor(image).float().cuda().permute(2, 0, 1) / 255.
        
        segmentation = np.array(copy.deepcopy(Image.open(f"./data/{cfg.train.seq}/seg/{filename}"))).astype(np.float32) # .replace('.jpg','.png')
        segmentation_tensor = torch.tensor(segmentation).float().cuda()
        segmentation_color = torch.stack((segmentation_tensor, torch.zeros_like(segmentation_tensor), 1 - segmentation_tensor))
        
        dataset_entries.append({'cam': camera, 'im': image_tensor, 'seg': segmentation_color,
                                'id': camera_id, 't': timestep, 'proj': opengl_proj})
    
    return dataset_entries


def initialize_batch_sampler(dataset:list[dict]) -> list[int]:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices


def get_data_point(batch_sampler:list[int], dataset:list[dict]) -> dict:
    if len(batch_sampler) < 1:
        batch_sampler = initialize_batch_sampler(dataset)
    return dataset[batch_sampler.pop()]


def initialize_params(cfg:DictConfig, priors:dict) -> tuple[dict, dict]:
    init_pt_cld:np.ndarray = priors['pt_cld']
    segmentation = init_pt_cld[:, 6]
    square_distance, _ = o3d_knn(init_pt_cld[:, :3], cfg.train.num_nearest)
    mean_square_distance = square_distance.mean(-1).clip(min=1e-7)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((segmentation, np.zeros_like(segmentation), 1 - segmentation), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (segmentation.shape[0], 1)),
        'logit_opacities': np.zeros((segmentation.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean_square_distance))[..., None], (1, 3)),
        'cam_m': np.zeros((cfg.train.max_cams, 3)),
        'cam_c': np.zeros((cfg.train.max_cams, 3)),
        'cam_pos': priors['w2c'][0, :, :3, 3], # Initial camera positions
        'cam_rot': build_quaternion(torch.tensor(priors['w2c'][0, :, :3, :3])), # Initial camera rotation matrices
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    scene_radius = priors['radius'] * cfg.train.scene_size_mult
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'prev_cam_pos': torch.tensor(priors['w2c'][0, ...]).cuda().float(),
                 'prev_cam_rot': torch.tensor(priors['w2c'][0, :, :3, :3]).cuda().float(),}
    schedulers['means3D'] = lr_scheduler(cfg.optimizer.means3D_lr * variables['scene_radius'],
                                         cfg.optimizer.means3D_lr * variables['scene_radius'] / 300.0,
                                         lr_delay_mult=cfg.optimizer.lr_delay_mult, max_steps=cfg.train.initial_timestep_iters)
    schedulers['rgb_colors'] = lr_scheduler(cfg.optimizer.means3D_lr * variables['scene_radius'],
                                         cfg.optimizer.means3D_lr * variables['scene_radius'] / 300.0,
                                         lr_delay_mult=cfg.optimizer.lr_delay_mult, max_steps=cfg.train.initial_timestep_iters)
    schedulers['cam_pos'] = lr_scheduler(cfg.optimizer.cam_pos_lr * variables['scene_radius'],
                                         cfg.optimizer.cam_pos_lr * variables['scene_radius'] / 300.0,
                                         lr_delay_mult=cfg.optimizer.lr_delay_mult, max_steps=cfg.train.initial_timestep_iters)
    schedulers['cam_rot'] = lr_scheduler(cfg.optimizer.cam_rot_lr * variables['scene_radius'],
                                         cfg.optimizer.cam_rot_lr * variables['scene_radius'] / 300.0,
                                         lr_delay_mult=cfg.optimizer.lr_delay_mult, max_steps=cfg.train.initial_timestep_iters)
    return params, variables


def initialize_optimizer(params:dict, variables:dict, cfg:DictConfig):
    lrs = {
        'means3D': cfg.optimizer.means3D_lr * variables['scene_radius'],
        'rgb_colors': cfg.optimizer.rgb_lr,
        'seg_colors': cfg.optimizer.seg_lr,
        'unnorm_rotations': cfg.optimizer.unnorm_rot_lr,
        'logit_opacities': cfg.optimizer.logit_opacity_lr,
        'log_scales': cfg.optimizer.log_scales_lr,
        'cam_m': cfg.optimizer.cam_m_lr,
        'cam_c': cfg.optimizer.cam_c_lr,
        'cam_pos': cfg.optimizer.cam_pos_lr * variables['scene_radius'],
        'cam_rot': cfg.optimizer.cam_rot_lr * variables['scene_radius'],
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)



def initialize_per_timestep(
    params: dict[str, torch.Tensor], 
    variables: dict[str, torch.Tensor], 
    optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    current_points = params['means3D']
    current_rotations_normalized = torch.nn.functional.normalize(params['unnorm_rotations'])

    # Calculate momentum-like updates
    new_points = current_points + (current_points - variables["prev_pts"])
    new_rotations = torch.nn.functional.normalize(current_rotations_normalized + (current_rotations_normalized - variables["prev_rot"]))

    # Extract foreground entities' info
    foreground_mask = params['seg_colors'][:, 0] > 0.5
    previous_inverse_rotations_foreground = current_rotations_normalized[foreground_mask]
    previous_inverse_rotations_foreground[:, 1:] = -1 * previous_inverse_rotations_foreground[:, 1:]
    foreground_points = current_points[foreground_mask]
    previous_offsets = foreground_points[variables["neighbor_indices"]] - foreground_points[:, None]

    # Update previous values in the variables dictionary
    variables['prev_inv_rot_fg'] = previous_inverse_rotations_foreground.detach()
    variables['prev_offset'] = previous_offsets.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = current_points.detach()
    variables["prev_rot"] = current_rotations_normalized.detach()
    variables["prev_pts"] = current_points.detach()
    variables["prev_rot"] = current_rotations_normalized.detach()

    # Update the params dictionary
    updated_params = {'means3D': new_points, 'unnorm_rotations': new_rotations}
    params = update_params_and_optimizer(updated_params, params, optimizer)
    # Update the params dictionary
    updated_params = {'means3D': new_points, 'unnorm_rotations': new_rotations}
    params = update_params_and_optimizer(updated_params, params, optimizer)

    return params, variables



def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()

    variables["prev_cam_pos"][..., :3, 3] = params['cam_pos'].detach()
    variables["prev_cam_pos"][..., :3, :3] = build_rotation(params['cam_rot'].detach())

    update_lr_per_iter(optimizer, 0)

    del schedulers['cam_pos']
    del schedulers['cam_rot']

    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c', 'cam_pos', 'cam_rot']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def update_lr_per_iter(optimizer, iter):
    params_to_update = ['means3D', 'cam_pos', 'cam_rot']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_update:
            param_group['lr'] = schedulers[param_group["name"]](iter)

def train(cfg : DictConfig):
    exp_name = cfg.train.exp
    sequence = cfg.train.seq
    if os.path.exists(f"./output/{exp_name}/{sequence}"):
        print(f"Experiment '{exp_name}' for sequence '{sequence}' already exists. Exiting.")
        return
   
    priors = get_priors(cfg)
    num_timesteps = len(priors['fn'])

    params, variables = initialize_params(cfg, priors)
    optimizer = initialize_optimizer(params, variables, cfg)
    output_params = []
    
    for timestep in range(num_timesteps):
        dataset = construct_timestep_dataset(timestep, priors, cfg)
        batch_sampler = initialize_batch_sampler(dataset)
        is_initial_timestep = (timestep == 0)
        if not is_initial_timestep:
            # "momentum-based update"
            # "momentum-based update"
            params, variables = initialize_per_timestep(params, variables, optimizer)
        
        num_iter_per_timestep = cfg.train.initial_timestep_iters if is_initial_timestep else cfg.train.timestep_iters
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {timestep}")

        if cfg.train.is_static: # If enabled, it will only perform the init. timestep training to visualize training.
            output_params.append(params2cpu(params, True))
        
        for i in range(num_iter_per_timestep):
            curr_data = get_data_point(batch_sampler, dataset)
            loss, variables = get_loss(cfg, params, curr_data, variables, is_initial_timestep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (cfg.train.is_static and i % (cfg.train.initial_timestep_iters / 100) == 0):
                output_params.append(params2cpu(params, False))
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (cfg.train.is_static and i % 50 == 0):
                output_params.append(params2cpu(params, False))
            
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep :
                    if cfg.train.fast_dynamics:
                        update_lr_per_iter(optimizer, i)
                    if cfg.sparsify.enabled:
                        params, variables = densify(params, variables, optimizer, i)
        
        progress_bar.close()
        if cfg.train.is_static:
            break
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
            
    save_params(output_params, sequence, exp_name)

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg : DictConfig):
    train(cfg)
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    main()