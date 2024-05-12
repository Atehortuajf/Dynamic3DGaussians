# This script generates a trimesh scene from the priors.
# Should do the same thing as scene.show() in the get_priors function, so this can be thought of as a test.
# This is straight up mostly yoinked from the dust3r source code.
import sys
import os
import trimesh
import torch
import numpy as np
import PIL.Image
sys.path.append('./dust3r')

from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam

def test_priors(cfg, priors):
    scene = trimesh.Scene()
    pts = priors['pt_cld'][:, :3]
    colors = priors['pt_cld'][:, 3:6]
    add_pointcloud(scene, pts, colors)

    for cam_idx in range(priors['w2c'].shape[1]):
        add_scene_cam(scene, np.linalg.inv(priors['w2c'][0, cam_idx]), priors['k'][0, cam_idx, 0, 0],
                      np.array(PIL.Image.open(os.path.join(cfg.data.path, f"ims/{priors['fn'][0][cam_idx]}"))),
                      priors['k'][0, cam_idx, 0, 0], 0.03)
        
    scene.show(line_settings={'point_size': 2})

def add_pointcloud(scene, pts3d, color, mask=None):
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)
        if mask is None:
            mask = [slice(None)] * len(pts3d)
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3))

        if isinstance(color, (list, np.ndarray, torch.Tensor)):
            color = to_numpy(color)
            col = np.concatenate([p[m] for p, m in zip(color, mask)])
            assert col.shape == pts.shape
            pct.visual.vertex_colors = uint8(col.reshape(-1, 3))
        else:
            assert len(color) == 3
            pct.visual.vertex_colors = np.broadcast_to(uint8(color), pts.shape)

        scene.add_geometry(pct)
        return scene

def uint8(colors):
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors *= 255
    assert 0 <= colors.min() and colors.max() < 256
    return np.uint8(colors)