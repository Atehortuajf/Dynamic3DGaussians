# Dynamic 3DGS expects the data for the priors to be in a very specific form.
# Instead of bloating the priors script with these peculiarities, this file
# exposes some functions to help abstract away the details.
import cv2
import os
import glob
import numpy as np
from omegaconf import DictConfig
from collections import defaultdict

# Processes the videos into jpg frames
def extract_frames(cfg : DictConfig):
    if os.path.exists(os.path.join(cfg.data.path, "ims")) and (not cfg.data.force_extract):
        print("Frames already extracted.")
        file_names, w, h = register_fn(cfg.data.path)
        return file_names, w, h
    fn = recursive_dict()
    cam_id = 0
    for videopath in glob.glob(os.path.join(cfg.data.path, "*.mp4")):
        cam = cv2.VideoCapture(videopath)
        if not cam.isOpened():
            print("Error opening video file")
            return

        path = os.path.join(cfg.data.path, f"ims/{cam_id}")
        seg_path = os.path.join(cfg.data.path, f"seg/{cam_id}")
        # Create a new directory for frames if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        # Frame counter
        frame_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            # Save frame as PNG in the new directory
            frame_path = os.path.join(path, f"{frame_count:06d}.jpg")
            if (cfg.train.resize):
                resize_width = cfg.train.resize_size
                resize_height = int(frame.shape[0] * (resize_width / frame.shape[1]))
                frame = cv2.resize(frame, (resize_width, resize_height))
            h, w = frame.shape[:2]
            cv2.imwrite(frame_path, frame)
            generate_seg(frame_path, frame)
            fn[frame_count][cam_id] = f"{cam_id}/{frame_count:06d}.jpg"
            frame_count += 1

        cam.release()
        cam_id += 1
        print(f"Extracted {frame_count} frames to {path} and {seg_path}")
    return fn, w, h

# To train a static scene from ims
def prepare_static(cfg : DictConfig):
    if os.path.exists(os.path.join(cfg.data.path, "ims")) and (not cfg.data.force_extract):
        print("Frames already extracted.")
        file_names, w, h = register_fn(cfg.data.path)
        return file_names, w, h
    paths = glob.glob(os.path.join(cfg.data.path, "*.jpg"))
    fn = recursive_dict()
    for image_path, cam_id in zip(paths, range(len(paths))):
        frame = cv2.imread(image_path)
        path = os.path.join(cfg.data.path, f"ims/{cam_id}")
        seg_path = os.path.join(cfg.data.path, f"seg/{cam_id}")
        ims_path = os.path.join(path, "000000.jpg")
        # Create a new directory for frames if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)
        if (cfg.train.resize):
                resize_width = cfg.train.resize_size
                resize_height = int(frame.shape[0] * (resize_width / frame.shape[1]))
                frame = cv2.resize(frame, (resize_width, resize_height))
        h, w = frame.shape[:2]
        fn[0][cam_id] = f"{cam_id}/000000.jpg"
        cv2.imwrite(ims_path, frame)
        generate_seg(ims_path, frame)
    return fn, w, h

# TODO: Find a way to generate binary segmentation masks
def generate_seg(frame_path, frame):
    h, w = frame.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask_path = frame_path.replace("ims", "seg")
    cv2.imwrite(mask_path, mask)
    return

def register_fn(path):
    fn = recursive_dict()
    w = None
    for cam_path in os.listdir(os.path.join(path, "ims")):
        cam_id = int(cam_path.split("_")[0])
        for frame_path in os.listdir(os.path.join(path, "ims", cam_path)):
            timestep = int(frame_path.split(".")[0])            
            fn[timestep][cam_id] = os.path.join(cam_path, frame_path)
            if (w is None): # Kinda dumb but it works
                h, w = cv2.imread(os.path.join(path, "ims", cam_path, frame_path)).shape[:2]
    return fn, w, h

def extract_timeframe(cfg : DictConfig, timestep):
    ims_path = os.path.join(cfg.data.path, "ims")
    timestep_image_paths = []
    for cam_id in range(len(os.listdir(ims_path))):
        im_path = os.path.join(ims_path, str(cam_id), f"{timestep:06d}.jpg")
        timestep_image_paths.append(im_path)
    return timestep_image_paths

def recursive_dict():
    return defaultdict(recursive_dict)