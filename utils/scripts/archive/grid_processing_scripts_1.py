# Developed in Uniform_Grid
# Efficient, numpy based methods:

import numpy  as np
import open3d as o3d

def get_unitary_3d_directions():
    return np.rollaxis(np.mgrid[-1:2:2, -1:2:2, -1:2:2], 0, 4).reshape(-1, 3)

def is_point_in_voxel(point, voxel_grid, explored_voxels=None):
    '''
        Check if there is a voxel in voxel_grid conitaining point
        returns True / False.
        If explored_voxels is not None, it returns unexplored voxels.
    '''
    voxel_id     = tuple(voxel_grid.get_voxel(point))
    voxel_center = voxel_grid.get_voxel_center_coordinate(idx = voxel_id)

    if (explored_voxels != None):
        if (voxel_id in explored_voxels):
            return True
        else:
            return False

    return (voxel_center != 0).sum() != 0

def get_neighbors_range(scale=0, dx=1, dy=1, dz=1):
    '''
    How much (many voxels) should one go in each of the eight directions.
    (to consider neighbors)
    Get np with directions in which the neighbors fall.
    retrun shape is (8, 8^(scale+1), 3)
    '''
    limit = 2**scale
    return np.rollaxis(np.mgrid[1:limit*dx+1, 1:limit*dy+1, 1:limit*dz+1], 0, 4).reshape((-1, 3))

def get_neighbor_voxel_centers(point, neighbor_ranges, voxel_size):
    '''
        Get np with voxel center points in which the neighbors fall.
        it assumes that point is a corner of a previous voxel.
        retrun shape is (8, 8^(scale+1), 3).
    '''
    directions          = np.rollaxis(np.mgrid[-1:2:2, -1:2:2, -1:2:2], 0, 4).reshape(-1, 3)
    neighbor_ends       = point + voxel_size * neighbor_ranges * directions[:, np.newaxis, :]

    cell_center_offsets = - directions / 2 * voxel_size

    neighbor_centers    = neighbor_ends + cell_center_offsets[:, np.newaxis, :]
    return neighbor_centers

def get_o3d_pcd_from_ply(ply_path="../datasets/teapot.ply", num_points=32, seed=1):
    '''load point cloud from .ply path:'''

    mesh       = o3d.io.read_triangle_mesh(ply_path)
    pcd        = mesh.sample_points_poisson_disk(number_of_points=num_points, seed=seed)
    scaled_pcd = scale_pcd(pcd)

    return scaled_pcd


get_dim_range   = lambda Dd: np.hstack((np.arange(-Dd // 2, 0), (np.arange(1, Dd // 2 + 1))))

def get_utility_nn_input(point, Dx, Dy, Dz, voxel_grid, scales, voxel_size, explored=None):
    '''Get neighborhood of point in voxel_grid.
    If explored is a set of indexes, then return only explored voxels.'''
    np_input        = np.zeros((len(scales), Dx, Dy, Dz))

    #neighbors in specific cube:
    neigbor_ranges = [get_neighbors_range(scale=i) for i in scales]

    Dx_range        = get_dim_range(Dx)
    Dy_range        = get_dim_range(Dy)
    Dz_range        = get_dim_range(Dz)

    # point = initial_point

    for scale in scales:
        for dx in Dx_range:
            for dy in Dy_range:
                for dz in Dz_range:
                    # dx, dy, dz relative to input point.
                    np_direction        = np.array([dx, dy, dz])#Vector direction
                    #all ends of cells in direction [dx, dy, dz], at scale "scale"
                    neighbor_ends       = point + neigbor_ranges[scale] * np_direction * voxel_size
                    cell_center_offsets = - np_direction / np.abs(np_direction) / 2 * voxel_size
                    #All Centers of neighbors in direction [dx, dy, dz] at scale "scale":
                    neighbor_centers    = neighbor_ends + cell_center_offsets#[:, np.newaxis, :]

                    for center in neighbor_centers:
                        if is_point_in_voxel(center, voxel_grid, explored_voxels=explored):
                            #
                            ds_idx = np.hstack((np.where(Dx_range==dx)\
                                                , np.where(Dy_range==dy)\
                                                , np.where(Dz_range==dz))).reshape(-1)
                            np_input[scale][ds_idx[0]][ds_idx[1]][ds_idx[2]] = 1 #+=1 if counting all cells?
                            break
    return np_input



def scale_pcd(pcd, new_center=None, new_scale=None):
    '''
        Scale pcd to new_center and new_scale. 
        Defaults are previous center and scale 1 / unit cube with same center as before.
    '''
    new_center = pcd.get_center() if new_center is None else new_center
    # new_scale  = 1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()) if new_scale is None else new_scale
    scaled_pcd =  pcd.scale(scale=1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=new_center)
    if new_scale is None:
        return scaled_pcd
    else:
        # fit to unit cube
        return scaled_pcd.scale(scale=new_scale, center=new_center)


def get_voxel_grid(pcd, voxel_size):
    '''Voxelize pcd with voxel_size.'''

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    return voxel_grid


def set_pcd_colors(pcd, np_colors=np.zeros((1,3))):
    '''np_colors should have shape (N, 3), N=pcd.shape[0]'''

    o3d_colors = o3d.utility.Vector3dVector(np_colors)
    pcd.colors = o3d_colors

    return pcd


#Methods to get be optimized:
