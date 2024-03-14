"""
Taken from https://github.com/maurock/Dynamic3DGaussians/blob/depth_diff/data_making/blender_script.py
Makes initial point cloud. Has to run within blender.
"""
import bpy
import math
import mathutils
from random import random, choice
import os
import json
import bmesh
import numpy as np
from mathutils import Matrix

########## SETUP VARIABLES #############################################
# These are the only variables you need to set.
render_engine = 'CYCLES'
resolution_x = 600
resolution_y = 400
cycles_samples = 100
num_cameras = 50
radius = 10

output_point_path = 'path/to/init_pt_cld.npz' # this is the folder where cameras info are stored, please call it 'init_pt_cld.npz'

# Specify the name of your Scene Collection
collection_name = "Collection 1"
n_points_to_sample = 3000  # per mesh
###########################################################################

######## DEFINE FUNCTIONS ##################################################

# Function to generate a random point on a triangle
def random_point_in_triangle(v1, v2, v3):
    r1, r2 = random(), random()
    sqrt_r1 = math.sqrt(r1)

    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = r2 * sqrt_r1

    return (u * v1.co) + (v * v2.co) + (w * v3.co)

# Function to generate a random point on a polygon
def random_point_in_polygon(verts):
    # Triangulate the polygon, the first vertex is shared by all triangles
    triangles = [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]

    # Choose a random triangle
    chosen_triangle = choice(triangles)

    # Generate a random point in the chosen triangle
    return random_point_in_triangle(*chosen_triangle)


# Function to get the material color
def get_material_color(obj, face):
    mat_color = [1, 1, 1]  # Default color (white) in case no material is found
    if obj.material_slots and face.material_index < len(obj.material_slots):
        mat = obj.material_slots[face.material_index].material
        if mat and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    base_color = node.inputs['Base Color'].default_value
                    mat_color = [base_color[0], base_color[1], base_color[2]]  # RGB values
                    break
    return mat_color

###### SAMPLE POINTS FROM MESH ###############################################################
# Ensure you're in object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Initialize lists to store points and colors
means3D = []
rgb_colors = []
seg = []

# Get the collection
collection = bpy.data.collections.get(collection_name)
if not collection:
    print(f"No collection found by the name '{collection_name}'")
else:
    # Iterate through each object in the collection
    for obj in collection.objects:
        if obj.type == 'MESH':
            # Create a BMesh from the object
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.transform(obj.matrix_world)
            
            # Update the BMesh's internal index table
            bm.faces.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
                
            # Sample points
            for _ in range(n_points_to_sample):  # replace 100 with the number of samples you want

                # Find a random face
                face = bm.faces[int(random() * len(bm.faces))]
                # Calculate a random point on the face
                # point = face.calc_center_median()
                # Check if the face is a triangle or a polygon
                if len(face.verts) == 3:
                    # If triangle, directly use the vertices
                    point = random_point_in_triangle(*face.verts)
                else:
                    # If polygon, first triangulate
                    point = random_point_in_polygon(face.verts)
                means3D.append(point.to_tuple())  # Store point
                
                print(obj.name, point)
                
                # Get the color of the material of the face
                mat_color = get_material_color(obj, face)
                rgb_colors.append(mat_color)  

            # Free the BMesh
            bm.free()

# Concatenate means3D, rgb_colors and seg
means3D = np.array(means3D)
rgb_colors = np.array(rgb_colors)
seg = np.ones(shape=(rgb_colors.shape[0], 1))
sampled_data = dict()
sampled_data['data'] = np.concatenate((means3D, rgb_colors, seg), axis=1)

# Save the dictionary as a .npz file
np.savez_compressed(output_point_path, **sampled_data)