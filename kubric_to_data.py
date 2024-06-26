
"""
Script to convert Kubric output data to the format required by the implementation. Run blender_script.py after for pt cloud
"""
import argparse
from collections import defaultdict
import json
import os
import re
import numpy as np
import shutil
from PIL import Image
from glob import glob

def get_intrinsics(metadata):
  # Extracting the necessary values from metadata
  focal_length_mm = metadata['camera']['focal_length']
  sensor_width_mm = metadata['camera']['sensor_width']
  resolution = metadata['metadata']['resolution']

  # Calculating the focal lengths in pixel units
  fx = fy = focal_length_mm * (resolution[0] / sensor_width_mm)

  # Calculating the principal point (center of the image)
  cx = resolution[0] / 2
  cy = resolution[1] / 2

  # Building the intrinsic matrix
  intrinsics_matrix = np.array([
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1]
  ])

  return intrinsics_matrix


def post_process_json(nested_dict):
    """
    Converts a nested dictionary with integer keys to a nested list.
    
    :param nested_dict: A dictionary with integer keys at both levels.
    :return: A nested list where each inner list corresponds to the second-level dictionary.
    """
    if not nested_dict:
        return []

    # Determine the maximum key for the outer level
    max_outer_key = max(nested_dict.keys())
    # Initialize the outer list
    outer_list = [None] * (max_outer_key + 1)

    # Populate the outer list with inner lists
    for outer_key in range(max_outer_key + 1):
        inner_dict = nested_dict.get(outer_key, {})
        # Determine the maximum key for the inner level
        max_inner_key = max(inner_dict.keys(), default=-1)
        # Convert the inner dictionary to a list
        inner_list = [inner_dict.get(inner_key) for inner_key in range(max_inner_key + 1)]
        outer_list[outer_key] = inner_list

    return outer_list

def populate_jsons(count, camera_num, camera_path, w2c, k, cam_id, fn):
    with open(os.path.join(camera_path, 'metadata.json'), 'r') as file:
            metadata = json.load(file)
    for frame_path in glob(os.path.join(camera_path, 'rgba_*.png')):
        frame_id = int(re.search(r'\d+(?=\.png)', frame_path).group())
        c2w = np.array(metadata['camera']['R'])
        #c2w[:3,:3] = coord_swap(c2w[:3,:3])
        w2c[frame_id][count] = np.linalg.inv(c2w).tolist()
        k[frame_id][count] = get_intrinsics(metadata).tolist()
        cam_id[frame_id][count] = camera_num
        fn[frame_id][count] = "{}/{}.png".format(camera_num, str(frame_id).zfill(6))
    return metadata

def main(args):
    cameras = glob(os.path.join(args.data_path, 'camera_*'))

    # Prepare ims and seg folder
    for camera_path in cameras:
        camera_id = re.search(r'(?<=camera_)\d+', camera_path).group()
        ims_folder = os.path.join(args.output_path, 'ims', camera_id)
        seg_folder = os.path.join(args.output_path, 'seg', camera_id)

        if not os.path.exists(ims_folder):
            os.makedirs(ims_folder)

        if not os.path.exists(seg_folder):
            os.makedirs(seg_folder)

        for img in glob(os.path.join(camera_path, 'rgba_*.png')):
            frame_id = re.search(r'\d+(?=\.png)', img).group()
            shutil.copy(os.path.join(camera_path, 'rgba_{}.png'.format(frame_id.zfill(5))),
                        os.path.join(ims_folder, '{}.png'.format(frame_id.zfill(6))))
        for seg in glob(os.path.join(camera_path, 'segmentation_*.png')):
            frame_id = re.search(r'\d+(?=\.png)', seg).group()
            seg = Image.open(os.path.join(camera_path, 'segmentation_{}.png'.format(frame_id.zfill(5)))).convert('1')
            seg.save(os.path.join(seg_folder, '{}.png'.format(frame_id.zfill(6))))

    # Prepare train/test json
    split = 5 # 1:X split in train/test, TODO make this an arg
    data = dict()
    data_t = dict()
    w2c, k, cam_id, fn = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    w2c_t, k_t, cam_id_t, fn_t = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)

    count_train = 0
    count_test = 0
    for camera_path in cameras:
        camera_id = int(re.search(r'(?<=camera_)\d+', camera_path).group())
        if camera_id % split == 0:
            metadata = populate_jsons(count_test, camera_id, camera_path, w2c_t, k_t, cam_id_t, fn_t)
            count_test += 1
        else:
            metadata = populate_jsons(count_train, camera_id, camera_path, w2c, k, cam_id, fn)
            count_train += 1

    # Train data
    data['w'] = metadata['metadata']['resolution'][0]
    data['h'] = metadata['metadata']['resolution'][1]
    data['w2c'] = post_process_json(w2c)
    data['k'] = post_process_json(k)
    data['cam_id'] = post_process_json(cam_id)
    data['fn'] = post_process_json(fn)

    # Test data
    data_t['w'] = metadata['metadata']['resolution'][0]
    data_t['h'] = metadata['metadata']['resolution'][1]
    data_t['w2c'] = post_process_json(w2c_t)
    data_t['k'] = post_process_json(k_t)
    data_t['cam_id'] = post_process_json(cam_id_t)
    data_t['fn'] = post_process_json(fn_t)
    
    with open(os.path.join(args.output_path, 'train_meta.json'), 'w') as f:
        json.dump(data, f)
    with open(os.path.join(args.output_path, 'test_meta.json'), 'w') as f:
        json.dump(data_t, f)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='', help='Path to the Kubric output data.')
    args.add_argument('--output_path', type=str, default='data/YOUR_DATASET', help='Path to the output data.')
    args = args.parse_args()
    main(args)