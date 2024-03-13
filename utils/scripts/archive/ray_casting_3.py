
from constants                 import Dx, Dy, Dz, SCALES
from skimage.draw              import line_nd
from tqdm                      import tqdm
from grid_processing_scripts_1 import get_unitary_3d_directions, get_utility_nn_input, is_point_in_voxel
from scene_loading_2           import view_field_o3d

import numpy   as np
import seaborn as sns


#Ray excluding the origin:
get_rayed_voxel_ids = lambda start, end: np.stack(line_nd(start, end, endpoint=True), axis=-1)[1:]

def get_limit_reaching_rays(Dx, Dy, Dz, maximum_scale):
    '''
        Returns a list of 3D rays from (0, 0, 0) to all edge voxels.
        Edge voxels for the cube 2**max_scale * (0:Dx, 0:Dy, 0:Dz).
        Each ray is  a list of indexes from 0 to an edge voxel id.
        Rays are computed using `skimage.draw.line_nd`.
    '''
    #Ray excluding the origin:
    #get_rayed_voxel_ids = lambda start, end: np.stack(line_nd(start, end, endpoint=True), axis=-1)[1:]

    #distance limits:
    dls              = [2 ** maximum_scale * d // 2 for d in (Dx, Dy, Dz)]
    #Get all limit idx's of the form (Dx, 0..Dy, 0..Dz) / (keep one limit fixed).
    vision_3d_limits = 1 + np.vstack((np.rollaxis(np.mgrid[:dls[0], :dls[1], dls[2]-1:dls[2]], 0, 4).reshape((-1, 3)),\
                                 np.rollaxis(np.mgrid[:dls[0], dls[1]-1:dls[1], :dls[2]], 0, 4).reshape((-1, 3)),\
                                 np.rollaxis(np.mgrid[dls[0]-1:dls[0], :dls[1], :dls[2]], 0, 4).reshape((-1, 3))
                                     ))#.shape
    rays = []
    for limit in vision_3d_limits:
        ray = get_rayed_voxel_ids(start=np.zeros(3), end=limit)
        rays.append(ray)
    return rays


RAYS = get_limit_reaching_rays(Dx, Dy, Dz, SCALES[-1])

def get_location_utility(point_coordinates, model, voxel_grid, explored=None):
    '''
        Predict utility of the the point (x, y, z)
        within voxel grid, using model.
    '''
    Dx, Dy, Dz, max_scale = model.input_shape[1:]
    Vs                    = voxel_grid.voxel_size
    nn_input              = get_utility_nn_input(point_coordinates\
                                    , Dx, Dy, Dz\
                                    , voxel_grid, range(max_scale)\
                                    , Vs, explored=explored)[np.newaxis,:]
    nn_input = np.moveaxis(nn_input, 1, -1) #Make Channels last, instead of first.

    return model.predict(nn_input, verbose=0)

def get_observed_voxels(point, voxel_grid, rays=RAYS, absolute_position=False, flatten=False):
    '''
        Computes Voxel Ids within `voxel_grid` observed from the origin `point`,
        using the `rays`.
        Returns the dictionary `observed_voxels` for each of the 8 directions
        ((-1, -1, -1) to (1, 1, 1)).
        If absolute_position is True, return indexes of voxels within scene
        , otherwise return indexes relative to the viewport.
        If flatten = True, return independent of direction.
    '''
    observed_voxels = {tuple(d) : set() for d in get_unitary_3d_directions()}
    voxel_size      = voxel_grid.voxel_size
    if is_point_in_voxel(point, voxel_grid):
        return observed_voxels

    # for ray in tqdm(rays):
    for ray in rays:
        for d in observed_voxels:
            for vid in ray:
                np_direction        = np.array(d)
                limit_ends          = point + vid * voxel_grid.voxel_size * np_direction
                cell_center_offset  = - np_direction / np.abs(np_direction) / 2 * voxel_size
                #All Centers of neighbors in direction [dx, dy, dz] at scale "scale":
                limit_center        = limit_ends + cell_center_offset#[:, np.newaxis, :]

                if is_point_in_voxel(limit_center, voxel_grid):
                    ob_vid = tuple(voxel_grid.get_voxel(limit_center) if absolute_position else vid)
                    observed_voxels[d].add(ob_vid)
                    break

    return {vid for vids in observed_voxels.values() for vid in vids} if flatten else observed_voxels

def get_painted_voxels(voxel_ids_list, colors, voxel_grid):
    '''Retuns a list of "voxel like" point clouds with:
    centers -> in the voxel_ids_list append
    colors  -> as in colors.
    Radius of painted voxels is 120% of the voxel_grid.voxel_size.
    '''
    painted_voxels = []
    for i, vid in enumerate(voxel_ids_list):
        v_center = voxel_grid.get_voxel_center_coordinate(vid)
        v_radius = voxel_grid.voxel_size / 2 + voxel_grid.voxel_size / 10

        painted_v = view_field_o3d(v_center, norm=np.inf, color = colors[i], npoints=1000, radius=v_radius)
        painted_voxels.append(painted_v)
    return painted_voxels


def paint_viewfield_voxels(point, voxel_grid, rays=RAYS, view_field_colors=sns.color_palette("husl", 8)[::-1]):
    '''Returns a list  of "voxel like" point_clouds simulating the view from 'point'.
    Paints accoridng to `observed_voxels` directions.
    Where observed_voxels is the output of get_observed_voxels(point, voxel_grid, rays).'''
    observed_voxels = get_observed_voxels(point, voxel_grid, rays)
    painted_grid    = []
    for i, d in enumerate(observed_voxels):
        if observed_voxels[d]:
            viewfield_point_list = point + (voxel_grid.voxel_size * np.array(d) * (np.vstack(observed_voxels[d])) \
                                            -  np.array(d) * .5 * voxel_grid.voxel_size)

            viewfield_idx_list = [voxel_grid.get_voxel(vc) for vc in viewfield_point_list]


            painted_grid += get_painted_voxels(viewfield_idx_list\
                               , colors = np.repeat([view_field_colors[i]]\
                                                    , len(observed_voxels[d]), axis=0)\
                               , voxel_grid=voxel_grid)
    return painted_grid
