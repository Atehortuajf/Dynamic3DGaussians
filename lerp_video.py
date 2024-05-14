import cv2
import numpy as np

from visualize import render
from helpers import params2rendervar

def lerp_video(params, priors):
    out_im = cv2.VideoWriter('rgb_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, (512, 384))

    poses = params['cam_pos']

    intrinsics_1 = priors['k'][0, 0, ...]
    intrinsics_2 = priors['k'][0, 1, ...]

    rendervar = params2rendervar(params)

    time = [float(i) / 60.0 for i in range(60)]
    for idx in range(poses.shape[0] - 1):
        for t in time:
            w2c_t = lerp(poses[idx], poses[idx + 1], t)
            intrinsics_t = lerp(intrinsics_1, intrinsics_2, t)
            image, _ = render(w2c_t, intrinsics_t, rendervar)
            image_bgr = cv2.cvtColor(np.uint8((image.clamp_min(0).permute(1,2,0)*255).cpu().numpy()), cv2.COLOR_RGB2BGR)
            out_im.write(image_bgr)
        out_im.release()

def lerp(a, b, t):
    return a * (1 - t) + b * t