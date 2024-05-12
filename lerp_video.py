import cv2
import numpy as np

from visualize import render
from helpers import params2rendervar

def lerp_video(params, priors):
    out_im = cv2.VideoWriter('rgb_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, (512, 384))

    w2c_1 = get_w2c(params['cam_pos'][0], params['cam_rot'][0])
    w2c_2 = get_w2c(params['cam_pos'][1], params['cam_rot'][1])

    intrinsics_1 = priors['k'][0, 0, ...]
    intrinsics_2 = priors['k'][0, 1, ...]

    rendervar = params2rendervar(params)

    time = [float(i) / 60.0 for i in range(60)]
    for t in time:
        w2c_t = lerp(w2c_1, w2c_2, t)
        intrinsics_t = lerp(intrinsics_1, intrinsics_2, t)
        image, _ = render(w2c_t, intrinsics_t, rendervar)
        image_bgr = cv2.cvtColor(np.uint8((image.clamp_min(0).permute(1,2,0)*255).cpu().numpy()), cv2.COLOR_RGB2BGR)
        out_im.write(image_bgr)
    out_im.release()
    print('done')


def get_w2c(pos, rot):
    w2c = np.eye(4)
    w2c[:3, :3] = rot.detach().cpu().numpy()
    w2c[:3, 3] = pos.detach().cpu().numpy()
    return w2c

def lerp(a, b, t):
    return a * (1 - t) + b * t