# Dynamic 3DGS expects the data for the priors to be in a very specific form.
# Instead of bloating the priors script with these peculiarities, this file
# exposes some functions to help abstract away the details.
import cv2
import os
import glob
from omegaconf import DictConfig
from collections import defaultdict

# Processes the videos into jpg frames
def extract_frames(cfg : DictConfig):
    if os.path.exists(os.path.join(cfg.data.path, "ims")) and (not cfg.data.force_extract):
        print("Frames already extracted.")
        file_names = register_fn(cfg.data.path)
        return file_names
    fn = recursive_dict()
    cam_id = 0
    for videopath in glob.glob(os.path.join(cfg.data.path, "*.mp4")):
        cam = cv2.VideoCapture(videopath)
        if not cam.isOpened():
            print("Error opening video file")
            return

        path = os.path.join(cfg.data.path, f"ims/{cam_id}")
        # Create a new directory for frames if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Frame counter
        frame_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            # Save frame as PNG in the new directory
            frame_path = os.path.join(path, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            fn[frame_count][cam_id] = frame_path
            frame_count += 1

        cam.release()
        cam_id += 1
        print(f"Extracted {frame_count} frames to {path}")

def register_fn(path):
    fn = recursive_dict()
    for cam_path in os.listdir(os.path.join(path, "ims")):
        cam_id = int(cam_path.split("_")[0])
        for frame_path in os.listdir(os.path.join(path, "ims", cam_path)):
            timestep = int(frame_path.split(".")[0])            
            fn[timestep][cam_id] = os.path.join(cam_path, frame_path)
    return fn

def extract_timeframe(cfg : DictConfig, timestep):
    ims_path = os.path.join(cfg.data.path, "ims")
    timestep_image_paths = []
    for cam_path in os.listdir(ims_path):
        im_path = os.path.join(ims_path, cam_path, f"{timestep:06d}.jpg")
        timestep_image_paths.append(im_path)
    return timestep_image_paths

def recursive_dict():
    return defaultdict(recursive_dict)