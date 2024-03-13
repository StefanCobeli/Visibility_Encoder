import numpy   as np
import pyvista as pv# Scene Rendering
import open3d  as o3d
# from grid_processing_scripts_1 import scale_pcd, o3d
# from tqdm import tqdm
# import time



def get_pyvista_mesh(ply_file_path, verbose=True):
    """Load pyvista mesh from ply file:"""
    mesh                  = pv.read(ply_file_path)
    num_points_full_scene = mesh.points.shape[0]
    if verbose:
        print(f"Scene {ply_file_path.split('/')[-1]} has: {mesh.points.shape[0]:,} points.")

    return mesh

def get_o3d_pcd_from_pyvista_mesh(mesh, sample_scene_size=None, scaling=False, verbose=True):
    """Return point cloud from the pyvista mesh.
    If sample_scene_size, returns a sample of points in the scene"""

    num_points_full_scene = mesh.points.shape[0]
    if sample_scene_size:
        np.random.seed(1)
        points_idx = np.random.choice(num_points_full_scene, size=sample_scene_size, replace=False)
    else:
        points_idx = np.arange(num_points_full_scene)
    if verbose:
        print(f"Open3D PCD cloud created with: {points_idx.shape[0]:,} points.")

    selected_points = mesh.points[points_idx]
    selected_colors = mesh.get_array("RGB")[points_idx]/256

    pcd_full        = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(selected_points)
    pcd_full.colors = o3d.utility.Vector3dVector(selected_colors)
    
    from grid_processing_scripts_1 import scale_pcd
    pcd             = scale_pcd(pcd_full) if scaling else pcd_full

    return pcd


def get_potential_locations_bounds(voxel_grid, step_size):
    """Number of potential locations on x, y and z, based on the step size and on the voxel grid."""
    voxel_size    = voxel_grid.voxel_size
    grid_bounds   = voxel_grid.get_axis_aligned_bounding_box()
    point_dirs    = ((grid_bounds.get_max_bound() - grid_bounds.get_min_bound()) // voxel_size) // step_size
    return point_dirs

def get_viewport_coordinates(voxel_grid, step_size, hard_coded_mins=[-.6,-.6,-.6], hard_coded_maxs=[0, 0, 0]#, hard_coded_mins = [-1, -1, -1]\
    , dataset_size=None):
    '''Return Potential Viewport Locations / Coordinates.'''

    # hard_coded_mins = [0, 0, 2] # width, length, height
    voxel_size    = voxel_grid.voxel_size
    #initial_point = voxel_grid.get_axis_aligned_bounding_box().get_min_bound() + 0 * abs(STEP_SIZE - 3) * VOXEL_SIZE
    initial_point = voxel_grid.get_axis_aligned_bounding_box().get_min_bound() + abs(step_size - 3) * voxel_size
    #How many steps in each direction can we pick
    # grid_bounds   = voxel_grid.get_axis_aligned_bounding_box()
    # point_dirs    = ((grid_bounds.get_max_bound() - grid_bounds.get_min_bound()) // voxel_size) // step_size #+ 1
    point_dirs    = get_potential_locations_bounds(voxel_grid, step_size) + np.array(hard_coded_maxs)

    print("Grid of viewports genereated:", point_dirs, np.prod(point_dirs))

    training_points_locations_full = []
    for x in np.arange(hard_coded_mins[0], point_dirs[0]):
        for y in np.arange(hard_coded_mins[1], point_dirs[1]):
            for z in np.arange(hard_coded_mins[2], point_dirs[2]):#hc_min[2] + n (e.g. -.6, .4, 1.4, ...)
                t_point = initial_point + np.array([x, y, z]) * voxel_size * step_size
                if voxel_grid.get_voxel_center_coordinate(voxel_grid.get_voxel(t_point)).sum() != 0:
                    continue
                training_points_locations_full.append(t_point)

    # if DATASET_SIZE > len(training_points_locations_full):
    if dataset_size:
        np.random.seed(1)
        random_ids                = np.random.choice(np.arange(len(training_points_locations_full))\
                                                     , dataset_size, replace=False)
        training_points_locations = np.array(training_points_locations_full)[random_ids]
    else:
        training_points_locations = np.array(training_points_locations_full)


    return training_points_locations


def get_o3d_pcd_from_coordinates(np_locations, colors=None):
    '''`np_locations` (and optional `colors` 0-1) are np array with shape (n, 3). Can be obtained by int rgb / 255.
    If color is 3-tuple, color is constant.'''
    pcd_locations        = o3d.geometry.PointCloud()
    pcd_locations.points = o3d.utility.Vector3dVector(np.array(np_locations))
    if not(colors is None):
        if len(colors) != np_locations.shape[0]:
            colors = np.repeat([colors], np_locations.shape[0], axis=0)
        pcd_locations.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd_locations

# https://stackoverflow.com/a/33977530/7136493
def sample_spherical(offset=np.zeros((1, 3)), npoints=1000, radius=0.002, ndim=3, norm=np.inf):
    vec = (2 * np.random.rand(ndim, npoints) )-1
    vec /= np.linalg.norm(vec, axis=0, ord=norm) #* diameter
#     vec = vec * 20
    vec = vec * radius#* VOXEL_SIZE * 2500
    vec = vec.T
    return vec + offset


# https://seaborn.pydata.org/tutorial/color_palettes.html
# view_field_colors = sns.color_palette("mako", n_colors=10)#[-2] = (0.4285828, 0.82635051, 0.6780564)
def view_field_o3d(highlight_point, norm=2, color=(0.4285828, 0.82635051, 0.6780564), npoints=100, radius=1):#0.002*64):
    color=(0.8509803921568627, 0.21568627450980393, 0.43137254901960786)#red happy hue
    view_field            = sample_spherical(offset=highlight_point, npoints=npoints, radius=radius, norm=norm)
    view_field_o3d        = o3d.geometry.PointCloud()
    view_field_o3d.points = o3d.utility.Vector3dVector(view_field)
    view_field_o3d.colors = o3d.utility.Vector3dVector(np.repeat([color]\
                                                                 , repeats=view_field.shape[0], axis=0))

    return view_field_o3d

#Copied from full_pipeline_360_scripts:
def get_highlighed_o3d_locations(point_locations, norm=2, color=(0.8509803921568627, 0.21568627450980393, 0.43137254901960786), npoints=100, radius=1):
    '''Generate Open3D point cloud spheres centered in specified locations.'''
    if len(point_locations.shape) == 1:
        return [view_field_o3d(point_locations, norm=norm, color=color, npoints=npoints, radius=radius)]

    spheres_o3d = [view_field_o3d(p, norm=norm, color=color, npoints=npoints, radius=radius) for p in point_locations]
    return spheres_o3d

# def view_field_o3d_from_np(viewfield_point_list, color=(0.4285828, 0.82635051, 0.6780564)):

#     view_field_o3d        = o3d.geometry.PointCloud()
#     view_field_o3d.points = o3d.utility.Vector3dVector(viewfield_point_list)
#     view_field_o3d.colors = o3d.utility.Vector3dVector(np.repeat([color]\
#                                                                  , repeats=viewfield_point_list.shape[0], axis=0))

#     return view_field_o3d
