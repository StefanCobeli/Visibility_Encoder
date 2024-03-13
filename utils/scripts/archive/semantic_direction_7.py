#Methods developed in notebooks:
# 1_Model_Utility_v2_with_Direction
# 2_Directional_Semantic_Distribution_Model_Training

'''
    Loading a point cloud scene and creating directional input for a NN utility model.
'''

from constants                      import * 
from constants_direction            import * 
from helper_ply_SensatUrban_0       import read_ply, DataProcessing, Plot


from scene_loading_2                import *
from directed_utility_inputs_4      import *
from selection_algorithm_5          import * #check_clear_neighborhood

from skimage.draw    import line_nd
from skimage.measure import block_reduce
from tqdm import tqdm 


max_depth            = DEPTHS[-1]
max_width            = get_max_width_from_depth(max_depth)
max_height           = get_max_height_from_depth(max_depth)
directed_n_ids       = get_directed_view_neighbors_indexes(max_width, max_height, max_depth, vision_limits=False)
relative_offset      = - directed_n_ids[0]
relative_id_to_index = lambda rel_ids:    np.abs(rel_ids + relative_offset).astype(int)
# global_index_from_id = lambda global_ids: np.abs(global_ids - relative_offset)
global_index_from_id = lambda global_ids: (global_ids - relative_offset).astype(int)


def compute_cummulative_coverage_from_order(order_ids, order_name, s_utility_values\
                                            , s_all_locations, s_directions, voxel_grid\
                                            , coverage_limit=.97, max_steps=MAX_STEPS):
    #order               = order_dict[order_name]
    """
        Copied / adapted from the code used to compute the ground truth in notebook:
        1_Model_Utility_v2_with_Direction
    """
    
    order               = order_ids
    cummulative_utility = []
    explored_voxels     = np.zeros(shape=(1, 3))
    p_bar               = tqdm(order)
    
    num_voxels = len(voxel_grid.get_voxels())
    voxel_size = voxel_grid.voxel_size
    
    for crt_idx in p_bar:
        p_util  = s_utility_values[crt_idx]
        #p_loc   = potential_locations[crt_idx//NUM_ROTATIONS]
        p_loc   = s_all_locations[crt_idx]
        p_dir   = s_directions[crt_idx]

        neighbors                  = get_directed_neighbors(
                                                viewport_coordinates    = p_loc\
                                                        , direction     = p_dir\
                                                        , depth         = max_depth\
                                                        , voxel_size    = voxel_size
                                                        , vision_limits = False)

        cuboid_neighbors     = neighbors.reshape((max_width, max_height, max_depth, 3))
        directed_n_ids       = get_directed_view_neighbors_indexes(max_width, max_height, max_depth, vision_limits=False)
        relative_offset      = - directed_n_ids[0]
        relative_id_to_index = lambda rel_ids:    np.abs(rel_ids + relative_offset).astype(int)

        directed_screen                 = get_directed_view_neighbors_indexes(max_width, max_height, max_depth, vision_limits=True)
        get_rayed_voxel_ids_from_origin = lambda end: np.stack(line_nd(np.zeros(3), end, endpoint=True), axis=-1)[1:]
        directed_rays                   = np.apply_along_axis(func1d=get_rayed_voxel_ids_from_origin, axis=1, arr=directed_screen)
        ray_length                      = directed_rays[0].shape[0]



        inside_grid_neighbors, total_unique_hits, unique_view_ids = get_unique_hits(neighbors, voxel_grid, max_width, max_height, max_depth, directed_rays)

        unique_voxels   = cuboid_neighbors[tuple(relative_id_to_index(unique_view_ids).T)]

        if unique_voxels.shape[0] == 0:
            cummulative_utility.append(explored_voxels.shape[0])
            continue

        unique_vids     = np.apply_along_axis(lambda loc: voxel_grid.get_voxel(loc), axis=1, arr=unique_voxels)
        explored_voxels = np.unique(np.vstack([explored_voxels, unique_vids]), axis=0)
        cummulative_utility.append(explored_voxels.shape[0])
        
        p_bar.set_description(f"{order_name}: {len(explored_voxels) / num_voxels * 100:.2f}% cummulative coverage...")
        if num_voxels * coverage_limit < len(explored_voxels) or (order==crt_idx).argmax() > max_steps:
            break
    
    return cummulative_utility, explored_voxels

def normalize_binary(np_binary):
    return 2 * np_binary - 1

def get_directional_nn_input(location, direction , voxel_grid, depth=DEPTHS[-1]):
    voxel_size            = voxel_grid.voxel_size
     
    max_depth            = depth
    max_width            = get_max_width_from_depth(max_depth)
    max_height           = get_max_height_from_depth(max_depth)
    directed_n_ids       = get_directed_view_neighbors_indexes(max_width, max_height, max_depth, vision_limits=False)
    relative_offset      = - directed_n_ids[0]
    input_neighbor_shape = (max_width//4, max_height//4, max_depth//4)
    
    neighbors             = get_directed_neighbors(
                                viewport_coordinates    = location\
                                        , direction     = direction\
                                        , depth         = depth\
                                        , voxel_size    = voxel_size
                                        , vision_limits = False)
    inside_grid_neighbors = np.array(check_np_coordinates_in_voxel_grid(neighbors, voxel_grid))
    inside_grid_neighbors = inside_grid_neighbors.reshape((max_width, max_height, max_depth))
    
    
    scale1_global_nids   = get_directed_view_neighbors_indexes(*input_neighbor_shape, vision_limits=False)
    r1_ids               = relative_id_to_index(scale1_global_nids)
    r2_ids               = relative_id_to_index(2 * scale1_global_nids)
    r3_ids               = relative_id_to_index(4 * scale1_global_nids)

    scale1_inputs = inside_grid_neighbors[tuple((r1_ids).T)].reshape(input_neighbor_shape)
    #scale1_neighbors = cuboid_neighbors[tuple((r1_ids).T)]
    scale2_inputs = inside_grid_neighbors[tuple((r2_ids).T)].reshape(input_neighbor_shape)
    scale3_inputs = inside_grid_neighbors[tuple((r3_ids).T)].reshape(input_neighbor_shape)

    return np.stack([scale1_inputs, scale2_inputs, scale3_inputs], axis=0) 




def get_class_specific_scene(class_indexes, xyz, rgb, labels, voxel_size=1):
    '''
        Create a point cloud and voxel grid from a list of cooridinates, colors and labels.
        `class_indexes` indicates which label values are to be taken into account.
    '''
    cs_ids                       = np.isin(labels, class_indexes)
    xyz_cs, rgb_cs, labels_cs    = xyz[cs_ids], rgb[cs_ids], labels[cs_ids] 
    
    
    rgb_cs_semantic              = Plot.get_rgb_from_urban_labels(labels_cs)

    pcd_cs_s                     = get_o3d_pcd_from_coordinates(xyz_cs, rgb_cs_semantic / 256)
    voxel_grid_cs                = get_voxel_grid(pcd=pcd_cs_s, voxel_size=voxel_size)
    
    return pcd_cs_s, voxel_grid_cs



def get_grid_locations(voxel_grid, step_size=6, collision_distance=COLLISION_DISTANCE\
                       , hard_coded_mins=[-3,-3,-.5], hard_coded_maxs=[3,3,3]):
    
    # grid_locations = get_viewport_coordinates(voxel_grid, step_size=10, hard_coded_mins=[-.6,-.6,-.6], dataset_size=None)
    grid_locations = get_viewport_coordinates(voxel_grid, step_size=step_size, hard_coded_mins=[-3,-3,-.5]\
                                              , hard_coded_maxs=[3,3,3], dataset_size=None)
    # grid_locations = get_viewport_coordinates(voxel_grid, step_size=60, hard_coded_mins=[-1,-1,-.5]\
    #                                           , hard_coded_maxs=[1,1,1], dataset_size=None)

    grid_locations.shape
    print("Computing potential locations, excluding collisions...")
    not_colliding      = []
    for p_loc in tqdm(grid_locations):
        not_colliding.append(check_clear_neighborhood_v2(p_loc, voxel_grid, voxel_range=collision_distance))    

    potential_locations = grid_locations[not_colliding]

    print(f"Filtered down to {potential_locations.shape[0]:,} / {grid_locations.shape[0]:,} locations without collision "
         f"within a {collision_distance} voxel radius.")
    
    return potential_locations



def get_unique_hits(neighbors, voxel_grid, max_width, max_height, max_depth, directed_rays):
    ########## 5. Check neighborhood centers in voxel grid;
    inside_grid_neighbors             = np.array(check_np_coordinates_in_voxel_grid(neighbors, voxel_grid))

    #if inside_grid_neighbors.sum() == 0:
    #    print("expected utilty is 0. Empty neighborhood.")
        #completly_empty_ids.append((i, crt_idx))
        #continue
    #inside_grid_neighbors_GT[crt_idx] = np.array([inside_grid_neighbors]).T
    #Needs to be prorformed right immediately after neighborhood carving.
    #Not enough memory to save neighborhhod and ray tracing values >50GB for 6k points.
    ########## 6. Perform ray tracing inside from 0,0,0 to each vision limits;
    #Concluding indexes pipeline:
    inside_grid_neighbors = inside_grid_neighbors.reshape((max_width, max_height, max_depth))

    #7.a. ray trace from 0 to directed screen;
    #Trace all rays at the same time instead of #for ray in directed_rays:
    #b. transform each ray into relative ccordinates;
    indexes_hitting_grid  = tuple(relative_id_to_index(np.concatenate(directed_rays)).T)
    #c. Check if relative coordinates are occupied in the GT dataset;
    traced_hits           = inside_grid_neighbors[indexes_hitting_grid]
    
    ray_length = directed_rays[0].shape[0]
    #d. Pool max value from each ray: / How many rays hit the grid:
    hitting_rays_mask     = block_reduce(traced_hits, (ray_length), np.max).astype(bool)
    #d'. Unique voxels hit by the rays:
    hitting_ray_ids = np.arange(traced_hits.shape[0]//ray_length)[hitting_rays_mask]
    first_hit_id    = (traced_hits.reshape(-1, ray_length).argmax(axis=1))[hitting_rays_mask]
    seen_ids        = directed_rays[hitting_ray_ids, first_hit_id, :]
    unique_view_ids = np.unique(seen_ids, axis=0)
    #unique_voxels   = cuboid_neighbors[tuple(relative_id_to_index(unique_view_ids).T)]
    total_unique_hits = unique_view_ids.shape[0]
    return inside_grid_neighbors, total_unique_hits, unique_view_ids


# Main idea Inspired from ChatGPT
# Alternative working idea can be found

# https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
import numpy as np
from scipy.spatial.distance import cdist

def get_second_order_in_original(order_2, order_1):
    """
        if array aranged on 'order_2' indexed on 'order_1',
        return order_2 in the original array.
        https://math.stackexchange.com/questions/549186/the-product-or-composition-of-permutation-groups-in-two-line-notation
        Test:
        np.random.seed(1)
        num_points          = 6
        mock_points         = np.random.randint(low=0, high=2, size=(num_points, 3))
        mock_utilities      = np.random.rand(num_points, 2)
        loc_1, order_1      = farthest_point_sampling(mock_points, 3, warm_ups=1)
        uti_2, order_2      = farthest_point_sampling(mock_utilities[order_1], 3, warm_ups=1)
        order_2_in_original = np.array(order_1)[order_2]
        np.equal(mock_utilities[order_2_in_original], uti_2)

    """
    return np.array(order_1)[order_2]

def farthest_point_sampling(points, k, warm_ups=5):
    """
        Return the first `k` points in order 
        The first `warm_ups` points are chose at random.
        If `wu==0` then, intial point is `points[0]`.

        Method generated using ChatGPT or the same can be found:
        # https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    """
    
    farthest_pts    = np.zeros((k, points.shape[1])) - 1
    np.random.seed(1)
    if warm_ups==0:
        point_ids       = np.array([0])
        farthest_pts[0] = points[0]
        warm_ups        = 1
    else:
        point_ids = np.random.choice(points.shape[0], size=warm_ups, replace=False)
        
    farthest_pts[:warm_ups] = points[point_ids]
    pts_ids                 = point_ids.tolist()
    dists = cdist(points, farthest_pts).min(axis=1)

    
    for i in range(warm_ups, k):
        # Choose the point with the largest distance to the previously selected points
        point_id        = np.argmax(dists)
        #if points[point_id] in farthest_pts:
        #    pts_ids = pts_ids + (k-len(pts_ids)) * [-1]
        #    break
        farthest_pts[i] = points[point_id]
        pts_ids.append(point_id)
        # Update the distances to the newly selected point
        dists = np.minimum(dists, cdist(points, [farthest_pts[i]]).ravel())
    
    return farthest_pts, np.array(pts_ids)


# np.array(farthest_point_sampling(full_features[:20], 5)).round(2)#.shape
# farthest_point_sampling(full_features_normed[:20], 5)#[0].round(2)


#Semantic color scheme by SensatUrban: can be found in helper_ply_SensatUrban.py
# ins_colors = [[85, 107, 47],    # 0. ground -> OliveDrab
#               [0, 255, 0],      # 1. tree -> Green
#               [255, 165, 0],    # 2. building -> orange
#               [41, 49, 101],    # 3. Walls ->  darkblue
#               [0, 0, 0],        # 4. Bridge -> black
#               [0, 0, 255],      # 5. parking -> blue
#               [255, 0, 255],    # 6. rail -> Magenta
#               [200, 200, 200],  # 7. traffic Roads ->  grey
#               [89, 47, 95],     # 8. Street Furniture  ->  DimGray
#               [255, 0, 0],      # 9. cars -> red
#               [255, 255, 0],    # 10. Footpath  ->  deeppink
#               [0, 255, 255],    # 11. bikes -> cyan
#               [0, 191, 255]     # 12. water ->  skyblue
#               ]

