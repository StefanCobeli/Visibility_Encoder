from constants                      import *
from grid_processing_scripts_1      import *
from scene_loading_2                import *
from ray_casting_3                  import *
# from helper_ply_SensatUrban_0       import *

from helper_ply_SensatUrban_0 import read_ply, DataProcessing, Plot

import os
import time





from keras.models import load_model
def main_pipeline_compute_potential_location_utilities(ply_file_name = PLY_FILE_NAME, scene_name=SCENE_NAME\
                                                       , step_size=STEP_SIZE, semantic=False):
    '''Full pipeline for location evaluation.'''
    start_time = time.time()
    # pcd, mesh, voxel_grid                  = get_processed_scene(ply_file_name)
    pcd, labels, voxel_grid                 = get_processed_scene_with_labels(ply_file_name, semantic)
    mt = time.time(); print(f"Scene {scene_name} voxelized in {mt - start_time:.2f} seconds.")
    np_utility_inputs, potential_locations = get_potential_locations_inputs(scene_name=scene_name, step_size=step_size\
                                                                            , voxel_grid=voxel_grid)
    print(f"Potential locations computed in {time.time() - mt:.2f} seconds."); mt = time.time()

    model = load_model(MODEL_ARCHITECTURE_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH)

    predicted_utilities = get_locations_utilities(model, np_utility_inputs)
    best_points         = get_utility_sorted_locations(potential_locations, predicted_utilities)
    
    print(f"Location visibility computed in {time.time() - mt:.2f} seconds.")
    print(f"Total location evaluation pipeline executed in {time.time() - start_time:.2f} seconds.")

    return pcd, labels, voxel_grid, model, predicted_utilities, best_points
    # return pcd, mesh, voxel_grid, model, predicted_utilities, best_points



def main_pipeline_compute_potential_location_utilities_directed(\
            ply_file_name = PLY_FILE_NAME, scene_name=SCENE_NAME\
            , step_size=STEP_SIZE   
        ):
    '''Full pipeline for location evaluation when considering directions.'''
    start_time = time.time()
    # pcd, mesh, voxel_grid                  = get_processed_scene(ply_file_name)
    pcd, labels, voxel_grid                 = get_processed_scene_with_labels(ply_file_name)
    mt = time.time(); print(f"Scene {scene_name} voxelized in {mt - start_time:.2f} seconds.")

    potential_locations = get_viewport_coordinates(voxel_grid=voxel_grid, step_size=step_size)
    # Code to be added from creating inputs in the 1_Model_Utility_v2_with_Direction notebook:
    # Maybe modify the method 'get_potential_locations_inputs' to generate directional inputs.
    # `np_utility_inputs` below 



    # np_utility_inputs, potential_locations = get_potential_locations_inputs(scene_name=scene_name, step_size=step_size\
    #                                                                         , voxel_grid=voxel_grid)
    # print(f"Potential locations computed in {time.time() - mt:.2f} seconds."); mt = time.time()

    # model = load_model(MODEL_ARCHITECTURE_PATH)
    # model.load_weights(MODEL_WEIGHTS_PATH)

    # predicted_utilities = get_locations_utilities(model, np_utility_inputs)
    # best_points         = get_utility_sorted_locations(potential_locations, predicted_utilities)
    
    # print(f"Location visibility computed in {time.time() - mt:.2f} seconds.")
    # print(f"Total location evaluation pipeline executed in {time.time() - start_time:.2f} seconds.")

    # return pcd, labels, voxel_grid, model, predicted_utilities, best_points
    # return pcd, mesh, voxel_grid, model, predicted_utilities, best_points



def display_open3d_geometries(geometries_list):
    '''Display open3d gemoertries list.'''
    # zoom=0.6, # front=[ -0.78, -0.081, 0.62 ], # lookat=pcd.get_center(),  # up=[ 0.64, -0.51, 0.77 ])
    o3d.visualization.draw_geometries(geometries_list)

def get_processed_scene(ply_file_path=PLY_FILE_NAME, voxel_size=VOXEL_SIZE, sample_scene_size=False, verbose=True):
    '''!!!!!DEPRECATED - changed to get_processed_scene_with_labels - to use SensatUrban starter code.
    Return point cloud, mesh and voxel grid according to the path and the voxel size.'''
    mesh       = get_pyvista_mesh(ply_file_path, verbose=verbose)
    pcd        = get_o3d_pcd_from_pyvista_mesh(mesh, sample_scene_size=sample_scene_size, verbose=verbose)
    voxel_grid = get_voxel_grid(pcd=pcd, voxel_size=voxel_size)
    
    return pcd, mesh, voxel_grid

# def get_pyvista_mesh(ply_file_path):
#     """Load pyvista mesh from ply file:"""
#     mesh                  = pv.read(ply_file_path)
#     num_points_full_scene = mesh.points.shape[0]
#     print(f"Scene {ply_file_path.split('/')[-1]} has: {mesh.points.shape[0]:,} points.")

#     return mesh


def get_scene_rectangle_dimenstions(voxel_grid):
    '''Return meter dimensions in each X, Y, Z directions.'''
    return (voxel_grid.get_max_bound() - voxel_grid.get_min_bound()) * voxel_grid.voxel_size

def get_processed_scene_with_labels(ply_file_path=PLY_FILE_NAME, voxel_size=VOXEL_SIZE, semantic=False, sem_and_rgb=False):
    '''
        Return point cloud, labels and voxel grid according to the path and the voxel size.
        If sem_and_rgb, return pcd, labels, voxel_grid, rgb
    '''
    #mesh       = get_pyvista_mesh(ply_file_path)
    #pcd              = get_o3d_pcd_from_pyvista_mesh(mesh, sample_scene_size=False)
    xyz, rgb, labels = DataProcessing.read_ply_data(ply_file_path) #to be surrounded by try except for case no labels
    
    if sem_and_rgb:
        semantic   = True
        rgb_colors = np.copy(rgb)

    if semantic:
        rgb = Plot.get_rgb_from_urban_labels(labels)
    

    num_points = labels.shape[0]
    print(f"Scene {ply_file_path.split('/')[-1]} has: \n\t{num_points:,} points.")

    pcd              = get_o3d_pcd_from_coordinates(xyz, rgb / 256)
    voxel_grid       = get_voxel_grid(pcd=pcd, voxel_size=voxel_size)
    num_voxels       = len(voxel_grid.get_voxels())
    print(f"Open3D Voxel Grid created with: \n\t{num_voxels:,} voxels; of size {voxel_size}.")
    
    xys_dims   = get_scene_rectangle_dimenstions(voxel_grid)
    print(f"Generated voxel grid for scene with dimensions:\n\t", xys_dims)
    print(f"Voxel density: \n\t{num_points / num_voxels:.2f} points per cube voxel.")
    print(f"Spatial density: \n\t{num_voxels / xys_dims.prod():.2f} space occupied inside cuboid.")
    
    if sem_and_rgb:
        return pcd, labels, voxel_grid, rgb_colors
    return pcd, labels, voxel_grid


def get_potential_locations_inputs(scene_name=SCENE_NAME, step_size=STEP_SIZE, np_files_path=NP_FILES_PATH, voxel_grid=None):
    '''Load potential location NN inputs and (x, y, z) coordinates'''
    
    pre_computed_path        = f"{np_files_path}/scenes/{scene_name}/"
    #!!!!!!!!!!Should load by Voxel Size as well:
    potential_locations_path = f"{pre_computed_path}/Grid_Locations_{scene_name}_Step_{step_size}.npy"
    utility_inputs_path      = f"{pre_computed_path}/Grid_Inputs_{scene_name}_Step_{step_size}.npy"
    
    try:
        potential_locations  = np.load(potential_locations_path)
        np_utility_inputs    = np.load(utility_inputs_path)
        print("Loaded pre-computed potential locations!")
        
    except:
        if voxel_grid:
            print("Computing pontential location details...")
        else:
            print("!!!!!!!!!!!!Provide method with a valid voxelized scene as Voxel Grid!!!!!!!!!!")
            return 
        if not os.path.exists(pre_computed_path): os.mkdir(pre_computed_path)
        potential_locations = get_viewport_coordinates(voxel_grid=voxel_grid, step_size=step_size)
        np_utility_inputs   = np.zeros(((len(potential_locations), len(SCALES), Dx, Dy, Dz)))

        for i in tqdm(range(potential_locations.shape[0])):
            np_utility_inputs[i] = get_utility_nn_input(point=potential_locations[i], Dx=Dx, Dy=Dy, Dz=Dz\
                             , voxel_grid=voxel_grid, scales=SCALES, voxel_size=VOXEL_SIZE)

        np_utility_inputs = np.moveaxis(np_utility_inputs, 1, -1)
        np.save(potential_locations_path, potential_locations)
        np.save(utility_inputs_path,      np_utility_inputs)
        
        print("Saved location details!")
    
    return np_utility_inputs, potential_locations
    

def get_highlighed_o3d_locations(point_locations, norm=2, color=[1, 0, 0], npoints=100, radius=VOXEL_SIZE):
    '''Generate Open3D point cloud spheres centered in specified locations.'''
    spheres_o3d = [view_field_o3d(p, norm=norm, color=color, npoints=npoints, radius=radius) for p in point_locations]    
    return spheres_o3d

def get_locations_utilities(model, utility_inputs):
    """Use Trained utility estimation model to predict visibility given occupancy maps input."""
    predicted_utilities = model.predict(utility_inputs)
    return predicted_utilities

def get_utility_sorted_locations(potential_locations, predicted_utilities, ascending=False):
    '''Get potential locations in sorted order based on their estimated utility.'''
    
    utility_sorted_ids = np.argsort(predicted_utilities, axis=0)[::-1]
    ordered_points     = potential_locations[utility_sorted_ids.T].reshape((-1, 3))
    
    return ordered_points[::-1] if ascending else ordered_points