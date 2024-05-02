# This file contains functions to help retrieve the required priors
# for the method (fg/bg segmentation, point cloud, camera params, etc.)
# using the Dust3r model.
import hydra
import numpy as np
import sys
import os
sys.path.append('./dust3r')
from omegaconf import DictConfig
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from data_preprocess import extract_frames, extract_timeframe

# TODO: Handle getting priors for arbitrary timesteps for mid scene reinitialization
@hydra.main(config_path="config", config_name="dust3r")
def get_priors(cfg : DictConfig):
    model = AsymmetricCroCo3DStereo.from_pretrained(cfg.dust3r.model_name).to(cfg.dust3r.device)
    fn = extract_frames(cfg)
    images = load_images(extract_timeframe(cfg, 0), size=512)
    pairs = make_pairs(images, scene_graph=cfg.dust3r.scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, cfg.dust3r.device, batch_size=cfg.dust3r.batch_size)

    scene = global_aligner(output, device=cfg.dust3r.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    _ = scene.compute_global_alignment(init="mst", niter=cfg.dust3r.niter, schedule=cfg.dust3r.schedule, lr=cfg.dust3r.lr)

    # retrieve priors from scene:
    imgs = np.array(scene.imgs)
    w, h = imgs.shape[1], imgs.shape[2]
    masks =  np.array([mask.float().detach().cpu().numpy() for mask in scene.get_masks()])
    intrinsics = get_intrinsics(scene, len(fn))
    w2c = (scene.get_im_poses().inverse()).detach().cpu().numpy()
    w2c = np.tile(w2c[None], (len(fn), 1, 1, 1))
    pts3d = np.array([im_pts.detach().cpu().numpy() for im_pts in scene.get_pts3d()])
    pt_cld = np.concatenate((pts3d, imgs, masks[..., None]), axis=-1).reshape(-1, 7)
    if (cfg.sparsify.enabled):
        pt_cld = sparsify(pt_cld, cfg.sparsify.num_samples)
    priors = {'imgs': imgs, 'masks': masks, 'k': intrinsics,
              'w2c': w2c, 'pt_cld': pt_cld, 'fn': fn, 'w': w, 'h': h}
    np.savez(cfg.data.priors, **priors)

    return priors

# Construct the intrinsics matrix from output
# TODO: We're currently assuming that intrinsics don't vary over time
def get_intrinsics(scene, num_timesteps):
    focals = scene.get_focals().detach().cpu().numpy().squeeze()
    pp = scene.get_principal_points().detach().cpu().numpy()
    intrinsics = np.zeros((focals.shape[0], 3, 3))
    intrinsics[:, 0, 0] = focals
    intrinsics[:, 1, 1] = focals
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]
    return np.tile(intrinsics[None], (num_timesteps, 1, 1, 1))

# Dust3r outputs a dense point cloud, but 3DGS works better with a sparse initialization
def sparsify(pts, num_samples):
    b = pts.shape[0]  # Number of points in pts
    if num_samples > b:
        raise ValueError("num_samples cannot be greater than the number of available points")
    # Randomly choose num_samples indices from range 0 to b-1 without replacement
    indices = np.random.choice(b, num_samples, replace=False)
    # Index pts with the chosen indices to get the sampled points
    sampled_pts = pts[indices]
    return sampled_pts

if __name__ == "__main__":
    get_priors()