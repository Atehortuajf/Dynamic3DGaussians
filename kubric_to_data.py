import argparse
from collections import defaultdict
import json
import os
import re
import numpy as np
import shutil
from PIL import Image
from glob import glob

def get_pt_cloud(camera_path):
    # Load depth image
    depth_image_path = os.path.join(camera_path, 'depth_00000.tiff')
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image)

    # Load RGB image
    rgb_image_path = os.path.join(camera_path, 'rgba_00000.png')
    rgb_image = Image.open(rgb_image_path)
    rgb_array = np.array(rgb_image)

    # Load metadata
    metadata_path = os.path.join(camera_path, 'metadata.json')
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    # Extract camera parameters from metadata
    fx = metadata['camera']['K'][0][0]
    fy = metadata['camera']['K'][1][1]
    cx = metadata['camera']['K'][0][2]
    cy = metadata['camera']['K'][1][2]

    # Generate point cloud from depth image
    height, width = depth_array.shape

    # Create a grid of coordinates corresponding to the image pixel coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Back-project the 2D pixel locations into 3D space
    z = depth_array
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # Stack the coordinates into a point cloud
    points = np.stack((x, y, z), axis=2)

    # Flatten the point cloud and remove points with zero depth
    max_depth_threshold = 10
    condition= np.logical_and(z > 0, z < max_depth_threshold)
    valid_points = points[condition].reshape(-1, 3)

    # Apply color to valid points
    valid_colors = rgb_array[condition].reshape(-1, 4)[:,:3] # Remove the alpha channel
    
    # Add segmentation (all ones for now)
    seg = np.ones(shape=(valid_colors.shape[0], 1))

    return np.concatenate((valid_points, valid_colors, seg), axis=1)

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
        w2c[frame_id][count] = metadata['camera']['R']
        k[frame_id][count] = metadata['camera']['K']
        cam_id[frame_id][count] = camera_num
        fn[frame_id][count] = "{}/{}.png".format(camera_num, str(frame_id).zfill(6))
    return metadata

def main(args):
    cameras = glob(os.path.join(args.data_path, 'camera_*'))

    # Initial point cloud
    pt_cloud = np.empty((0, 7))
    for camera_path in cameras:
        cam_cloud = get_pt_cloud(camera_path)
        pt_cloud = np.concatenate((pt_cloud, cam_cloud), axis=0)
    init_pt_cld = dict()
    init_pt_cld['data'] = pt_cloud
    # Save the dictionary as a .npz file
    np.savez_compressed(os.path.join(args.output_path, 'init_pt_cld.npz'), **init_pt_cld)

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
            # TODO might have to make it black and white?
            shutil.copy(os.path.join(camera_path, 'segmentation_{}.png'.format(frame_id.zfill(5))),
                        os.path.join(seg_folder, '{}.png'.format(frame_id.zfill(6))))

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