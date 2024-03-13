from grid_processing_scripts_1 import *

#Method taken from Next_Best_view_w_RT.ipynb notebook:

def check_clear_neighborhood(point, voxel_grid, voxel_range=1):
    '''Check if there is any collision in a ball of radius voxel_range.'''
    directions = get_unitary_3d_directions()
    ranges     = get_neighbors_range(0, voxel_range, voxel_range, voxel_range)
    voxel_size = voxel_grid.voxel_size
    
    #centers=[]
    for d in directions:
        for r in ranges:
            neighbor_ends       = point + d * r * voxel_size
            cell_center_offsets = - d / 2 * voxel_size 
            neighbor_center     = neighbor_ends + cell_center_offsets#[:, np.newaxis, :]
            #print(neighbor_center)
            #centers.append(neighbor_center)

            if is_point_in_voxel(neighbor_center, voxel_grid):
                return False#, centers
    return True and not(is_point_in_voxel(point, voxel_grid))#, centers#, centers



#can definetly be improved by using a code similar to the following:
# just need to define well directed_neighbors
# voxel_size           = voxel_grid.voxel_size
# rx, ry, rz           = voxel_range*2, voxel_range*2, voxel_range*2
# neighbor_ids         = get_neighbors_range(0, rx, ry, rz) - voxel_range #[-r...+r]
# neighbor_coordinates = point + neighbor_ids * voxel_size
# o3d_queries          = o3d.utility.Vector3dVector(neighbor_coordinates)
# inside_scene         = voxel_grid.check_if_included(o3d_queries)
# return np.sum(inside_scene) == 0

# This is more than 10 times faster compared to v1:
def check_clear_neighborhood_v2(point, voxel_grid, voxel_range=1):
    '''Check if there is any collision in a ball of radius voxel_range.'''
    voxel_size           = voxel_grid.voxel_size
    rx, ry, rz           = voxel_range*2, voxel_range*2, voxel_range*2
    neighbor_ids         = get_neighbors_range(0, rx, ry, rz) - voxel_range #[-r...+r]
    neighbor_coordinates = point + neighbor_ids * voxel_size
    inside_scene         = check_np_coordinates_in_voxel_grid(neighbor_coordinates, voxel_grid)
    return np.sum(inside_scene) == 0

def check_np_coordinates_in_voxel_grid(np_coordinates, voxel_grid):
    '''Given (n, 3) shaped np array and voxel grid, check if points are in grid.
    returns list of True / False for each position.'''
    o3d_queries          = o3d.utility.Vector3dVector(np_coordinates)
    inside_scene         = voxel_grid.check_if_included(o3d_queries)
    return inside_scene

########## or filter out points that are interesceting within 1 voxel to grid:
# o3d_queries         = o3d.utility.Vector3dVector(potential_locations)
# not_colliding ss      = np.logical_not(np.array(voxel_grid.check_if_included(o3d_queries)))
# not_colliding      = []
# COLLISION_DISTANCE = 5
# for p_loc in tqdm(potential_locations):
#     not_colliding.append(check_clear_neighborhood(p_loc, voxel_grid, voxel_range=COLLISION_DISTANCE))

# print("Computing potential locations, excluding collisions V2 (10-20xfaster compared to v1 above)...")