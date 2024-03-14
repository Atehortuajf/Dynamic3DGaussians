import os
import json
import random
import copy

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from transformers import pipeline

from helpers import setup_camera
from helpers import o3d_knn
from helpers import params2rendervar
from helpers import params2cpu
from helpers import save_params
from helpers import union_masks
from helpers import sample_pts_uniformly
from helpers import record_video

from external import calc_psnr
from external import densify
from external import update_params_and_optimizer

from loss import get_loss

MAX_CAMS:int =          50
NUM_NEAREST_NEIGH:int = 3
SCENE_SIZE_MULT:float = 1.1
USING_OBJ_REINIT:bool = True
RECORD_VID:bool = True

# Camera
NEAR:float =    1.0
FAR:float =     100.

# Training Hyperparams
INITIAL_TIMESTEP_ITERATIONS =   10_000
TIMESTEP_ITERATIONS =           2_000
GAUSS_REINIT_ITERATIONS =       5000


def construct_timestep_dataset(timestep:int, metadata:dict, sequence:str, mask_generator:SamAutomaticMaskGenerator) -> list[dict]:
    dataset_entries = []
    for camera_id in tqdm(range(len(metadata['fn'][timestep])), desc=f"Constructing dataset for timestep {timestep}"):
        width, height, intrinsics, extrinsics = metadata['w'], metadata['h'], metadata['k'][timestep][camera_id], metadata['w2c'][timestep][camera_id]
        camera = setup_camera(width, height, intrinsics, extrinsics, near=NEAR, far=FAR)
        
        filename = metadata['fn'][timestep][camera_id]
        
        image = np.array(copy.deepcopy(Image.open(f"./data/{sequence}/ims/{filename}")))[..., :3]
        image_tensor = (torch.tensor(image).float().cuda().permute(2, 0, 1) / 255.)

        if USING_OBJ_REINIT and (
            (timestep > 0) and (camera_id % 2 == 0)): # No need to generate masks for the first timestep since no new objects are added
            masks = mask_generator.generate(image)
        else:
            masks = []
        
        segmentation = np.array(copy.deepcopy(Image.open(f"./data/{sequence}/seg/{filename.replace('.jpg', '.png')}"))).astype(np.float32)
        segmentation_tensor = torch.tensor(segmentation).float().cuda()
        segmentation_color = torch.stack((segmentation_tensor, torch.zeros_like(segmentation_tensor), 1 - segmentation_tensor))
        
        dataset_entries.append({'cam': camera, 'im': image_tensor, 'seg': segmentation_color, 'id': camera_id, 'masks': masks, 'intrinsics': torch.tensor(intrinsics).cuda(), 'w2c': torch.tensor(extrinsics).cuda()})
    return dataset_entries


def check_obj_masks(params, variables, optimizer, dataset, depth_pipe, force = False):
    if not USING_OBJ_REINIT:
        return params, variables, False
    reinit_cams = []
    for entry in dataset:
        reinit_masks = []
        psnr_losses = []

        rendervar = params2rendervar(params)
        rendervar['means2D'].retain_grad()
        render, radius, _, = Renderer(raster_settings=entry['cam'])(**rendervar)

        target = entry['im']

        # Go through each mask and calculate PSNR
        for mask in entry['masks']:
            segmentation = torch.Tensor(mask['segmentation']).cuda()
            if segmentation.mean() < 0.02:
                continue # Skip masks that comprise less than 2% of the image
            segmentation = segmentation.bool()[None].repeat(3, 1, 1) # To use as indices
            masked_render = render[segmentation]
            masked_target = target[segmentation]
            psnr = calc_psnr(masked_render, masked_target).mean()
            psnr_losses.append((psnr, segmentation))
        
        # Calculate mean and standard deviation of PSNR losses
        psnr_values = torch.Tensor([loss[0] for loss in psnr_losses])
        mean_psnr = psnr_values.mean()
        stddev_psnr = psnr_values.std()
        variance_psnr = stddev_psnr ** 2

        if variance_psnr < 2:
            break # No need to reinitialize if variance is too low

        # Threshold for identifying low PSNR masks
        threshold = mean_psnr - 2 * stddev_psnr

        psnr_min = mean_psnr
        # Identify masks below the threshold, these should be where new objects appear
        for psnr, mask in psnr_losses:
            if psnr < threshold:
                reinit_masks.append((mask, psnr))
            if psnr < psnr_min:
                psnr_min = psnr
                mask_min = mask
        if force:
            reinit_masks.append((mask_min, psnr_min)) 
        if len(reinit_masks) > 0:
            reinit_cams.append((entry, reinit_masks))

    if len(reinit_cams) > 0:
        params, variables = initialize_new_objects(params, variables, optimizer, reinit_cams, depth_pipe)
    return params, variables, (len(reinit_cams) > 0)

def initialize_new_objects(params, variables, optimizer, reinit_cams, depth_pipe):
    print("New objects detected. Adding to scene.")
    num_means = len(params['means3D'])
    num_cams = float(len(reinit_cams))
    # For each camera with new objects
    for entry, masks in reinit_cams:
        # Get union mask and its complement
        u_mask = union_masks(masks)
        complement = ~union_masks(masks)

        # Do monocular depth estimation (scale invariant) TODO: fix this pil <-> tensor nonsense
        toPIL = T.ToPILImage()
        depth_est = T.functional.pil_to_tensor(depth_pipe(toPIL(entry['im']))['depth']).squeeze().float().cuda()
        depth_est = -depth_est + 255.0

        # Construct depth map mean from rendered gaussians
        rendervar = params2rendervar(params)
        image, radius, depth = Renderer(raster_settings=entry['cam'])(**rendervar)
        mean_depth_render = depth.squeeze()[complement].mean()

        # Get depth means at complement
        mean_depth_est = depth_est[complement].mean()

        # Scale depth estimation to match sparse depth
        scale = mean_depth_render / mean_depth_est

        # Apply scale to estimation
        depth_est = scale * depth_est

        # Sample new points from depth estimation restricted to union mask (as in paper)
        num_samples = num_means * (2/num_cams) # * u_mask.float().mean() this is currently broken
        num_samples = int(num_samples)
        new_means, new_colors, new_seg = sample_pts_uniformly(entry, depth_est, u_mask, num_samples)

        # Intialize new params
        new_params = {}
        # Initialize the new points in the same manner as the existing params
        square_distance, indices = o3d_knn(new_means.detach().cpu().numpy(), NUM_NEAREST_NEIGH)
        mean_square_distance = square_distance.mean(-1).clip(min=1e-7)

        new_params['means3D'] = new_means
        new_params['rgb_colors'] = new_colors.transpose(0,1)
        new_params['seg_colors'] = torch.stack((new_seg, torch.zeros_like(new_seg), 1 - new_seg), -1)
        new_params['unnorm_rotations'] = np.tile([1, 0, 0, 0], (new_seg.shape[0], 1))
        new_params['logit_opacities'] = np.zeros((new_seg.shape[0], 1))
        new_params['log_scales'] = np.tile(np.log(np.sqrt(mean_square_distance))[..., None], (1, 3))

        new_params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                    for k, v in new_params.items()}
        
        # Merge with existing params
        merged_params = params.copy()
        for k, v in new_params.items():
            merged_params[k] = torch.cat((params[k], v), 0)

        # Update params with optimizer
        params = update_params_and_optimizer(merged_params, params, optimizer)

        # Update the variables with the new points
        variables['max_2D_radius'] = torch.cat((variables['max_2D_radius'], torch.zeros(new_params['means3D'].shape[0]).cuda().float()))
        variables['means2D_gradient_accum'] = torch.cat((variables['means2D_gradient_accum'], torch.zeros(new_params['means3D'].shape[0]).cuda().float()))
        variables['denom'] = torch.cat((variables['denom'], torch.zeros(new_params['means3D'].shape[0]).cuda().float()))
    variables = initialize_post_first_timestep(params, variables, optimizer, num_knn=20, params_to_fix=['cam_m', 'cam_c'])
    return params, variables

def initialize_batch_sampler(dataset:list[dict]) -> list[int]:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices


def get_data_point(batch_sampler:list[int], dataset:list[dict]) -> dict:
    if len(batch_sampler) < 1:
        batch_sampler = initialize_batch_sampler(dataset)
    return dataset[batch_sampler.pop()]


def initialize_params(sequence:str, metadata:dict) -> tuple[dict, dict]:
    init_pt_cld:np.ndarray = np.load(f"./data/{sequence}/init_pt_cld.npz")["data"]
    segmentation = init_pt_cld[:, 6]
    square_distance, _ = o3d_knn(init_pt_cld[:, :3], NUM_NEAREST_NEIGH)
    mean_square_distance = square_distance.mean(-1).clip(min=1e-7)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((segmentation, np.zeros_like(segmentation), 1 - segmentation), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (segmentation.shape[0], 1)),
        'logit_opacities': np.zeros((segmentation.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean_square_distance))[..., None], (1, 3)),
        'cam_m': np.zeros((MAX_CAMS, 3)),
        'cam_c': np.zeros((MAX_CAMS, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(metadata['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = SCENE_SIZE_MULT * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params:dict, variables:dict):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
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

    # Update the params dictionary
    updated_params = {'means3D': new_points, 'unnorm_rotations': new_rotations}
    params = update_params_and_optimizer(updated_params, params, optimizer)

    return params, variables



def initialize_post_first_timestep(params, variables, optimizer, num_knn=20,
                                   params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']):
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


def train(sequence:str, exp_name:str):
    if os.path.exists(f"./output/{exp_name}/{sequence}"):
        print(f"Experiment '{exp_name}' for sequence '{sequence}' already exists. Exiting.")
        return
   
    vid_frames = []
    metadata = json.load(open(f"./data/{sequence}/train_meta.json", 'r'))
    num_timesteps = len(metadata['fn'])

    print('Loading SAM and Depth models')
    sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
    sam.cuda()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.96,
        box_nms_thresh=0.65
        )

    depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device='cuda')
    print('Done!')

    params, variables = initialize_params(sequence, metadata)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    
    for timestep in range(num_timesteps):
        new_objs = False
        dataset = construct_timestep_dataset(timestep, metadata, sequence, mask_generator)
        batch_sampler = initialize_batch_sampler(dataset)
        is_initial_timestep = (timestep == 0)
        if not is_initial_timestep:
            # "momentum-based update"
            params, variables = initialize_per_timestep(params, variables, optimizer)

            # Check for new objects and add them to scene if needed
            params, variables, new_objs = check_obj_masks(params, variables, optimizer, dataset, depth_pipe, (timestep == 14))
            
        num_iter_per_timestep = INITIAL_TIMESTEP_ITERATIONS if is_initial_timestep else (
            TIMESTEP_ITERATIONS if not new_objs else GAUSS_REINIT_ITERATIONS)
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {timestep}")
        
        for i in range(num_iter_per_timestep):
            curr_data = get_data_point(batch_sampler, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep, new_objs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep or new_objs:
                    params, variables = densify(params, variables, optimizer, i, new_objs)
        
        progress_bar.close()
        if RECORD_VID:
            im, _, _, = Renderer(raster_settings=dataset[0]['cam'])(**params2rendervar(params))
            vid_frames.append(im)
            
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep or new_objs:
            variables = initialize_post_first_timestep(params, variables, optimizer)
            
    save_params(output_params, sequence, exp_name)
    if RECORD_VID:
        record_video(vid_frames, sequence, exp_name)


def main():
    exp_name = "reinit"
    for sequence in ["adv"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    main()