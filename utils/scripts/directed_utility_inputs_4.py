#### Metohds to be docmuented.
### Mainly debelped in View_Matrix_to_World_Matrix.ipynb Notebook


# from grid_processing_scripts_1      import *
# from scene_loading_2                import *
# from ray_casting_3                  import *

# from constants                      import *

VOXEL_SIZE = 1
from scipy.spatial.transform import Rotation as R
import numpy as np
import seaborn as sns

def get_distance_based_colors(origin, neighbors, measure_unit=1):
    '''get colors for each `neighbor` based on the distance to the `origin`'''
    
    point_dists       = np.linalg.norm((origin - neighbors)/measure_unit, axis=1)
    point_dep_pallete = sns.color_palette("Spectral", n_colors=int(point_dists.max().round()+1))
    point_dep_colors  = np.take(point_dep_pallete, point_dists.astype(int), axis=0)

    return point_dep_colors




def get_vision_field_center(origin, max_x, max_y, max_z):
    return origin + max_z



# compute inverted up and front


def get_rotated_coordinates(xyz, scipy_model_rotation):# original_origin=if needed translation before rotation
    '''Given rotation xyz coordinates array and scipy rotation, 
    return the multiplication between the rotation matrix and xyz.'''
    rotation_matrix = scipy_model_rotation.as_matrix()
    # rotated_xyz     = np.matmul(rotation_matrix, (xyz-original_origin).T).T + original_origin
    rotated_xyz     = np.matmul(rotation_matrix, (xyz).T).T
    
    return rotated_xyz

def get_random_numerical_xyz_angles(num_locations=1):
    '''return xyz angles '''
    x_angle = 240 + 60 * np.random.random(num_locations)#260#90#270#360 * np.random.rand(1)[0] # Between -30 and 30
    y_angle = 180 * np.ones(num_locations) #360  * np.random.rand(1)[0]      # Should be constant
    z_angle = 360 * np.random.random(num_locations)#20#120#45 * np.random.rand(1)[0]      # 360
    return np.vstack([x_angle, y_angle, z_angle]).astype(int).T

def get_rotation_angles(custom_direction=None, random_direction=True):
    '''Get a scipy rotation based on custom_direction a list of three angles [x, y, z].
    Or get a random scipy rotation with appropriate human vision angles:
        y angle fixed to 180                - represents sidewise head rotation (would cause head tilting / nausea);
        z angle is any value from 0 to 360  - represents bodywise turning around;
        x angle is 270 + [-30, ..., 30]     - represents vertically raising or lowering head. 
    '''
    #return R.from_euler('xyz', [0, 0, 0], degrees=True)
    # return R.from_euler('xyz', [0, 0, np.pi/2], degrees=False)
    #np.random.seed(1)
    if custom_direction is None:
        x_angle, y_angle, z_angle = get_random_numerical_xyz_angles(1)
    # if random_direction:
        #x_angle = 240 + 60 * np.random.rand(1)[0]#260#90#270#360 * np.random.rand(1)[0] # Between -30 and 30
        #####x_angle = 240 + 40 * np.random.rand(1)[0]#260#90#270#360 * np.random.rand(1)[0] # Between -30 and 20
        #y_angle = 180 #360  * np.random.rand(1)[0]      # Should be constant
        #z_angle = 360 * np.random.rand(1)[0]#20#120#45 * np.random.rand(1)[0]      # 360
        ### print(x_angle, y_angle, z_angle)
        return R.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=True)
    else:
        #return R.from_euler('xyz', [0, 0, np.pi/2], degrees=False)
        # return R.from_euler('xyz', [3/2*np.pi, np.pi, np.pi + np.pi/4], degrees=False)
        # return R.from_euler('xyz', [3/2*np.pi, np.pi, np.pi - np.pi/3], degrees=False)
        #return R.from_euler('xyz', custom_direction, degrees=False)
        return R.from_euler('xyz', custom_direction, degrees=True)
        

def get_rotated_up_front_lookat(origin, scipy_model_rotation, lookat=-np.array([0,0,16])):
    '''Get open3d camera intialization parameters from a scipy rotation of the model around a given origin.'''
    up            = np.array([0,1,0])
    front         = np.array([0,0,1])
    #lookat        = -np.array([0,0,DEPTH])
    # lookat        = -np.array([0,0,32])
    #lookat        = -np.array([0,0,16])
    # lookat        = np.array([0,0,0])
    #lookat        = np.array([0,0,0]) # Does not permit additional zoom.
    #lookat should be towards the viewfield to permit zoom.
    
    #inverted_view = np.linalg.inv(scipy_model_rotation.as_matrix())
    inverted_view = scipy_model_rotation.as_matrix()
    r_up, r_front = np.matmul(inverted_view, np.vstack([up, front]).T).T
    r_lookat      = np.matmul(inverted_view, lookat.T).T + origin
    # r_lookat      = np.matmul(scipy_model_rotation.as_matrix(), lookat.T).T + origin
    
    return r_up, r_front, r_lookat 


############## COmputing MAX HEIGHT AND DEPTH accoring to: #############################
# https://akanoodles.medium.com/fui-yes-fui-7802862b1e01
# http://www.pixelsham.com/2019/11/20/colorwhat-is-the-resolution-of-the-human-eye/
# SOURCE: HUMAN ANATOMY & PHYSIOLOGY - PEARSON | DESIGN: NURAL CHOUDHURY - AKA NOODLES
# Source: Samsung Research Gear VR Horizontal / Veritical Field of View 
# // Binocular - Stereoscopic
# RIGHT EYE COVERAGE / # LEFT EYE COVERAGE / PERIPHERAL # MONOCULAR # TEMPORAL WITH EYEBALL
## Compuation for height and width based on provided angles and basic trigonometry 
# The only needed imput parameter is depth. In the medium article mentioned above,
# according to Samsung Research depth should be 11m for real 3D in VR.
# We fixed the intial depth to 16m. 
# We slightly adjsuted the W and H computation 
# based on empirical observations made on the Open3D visualizer.
################!!!ALL NUMBERS ARE APPROXIMATE VALUES!!!#############################

#Compute maximum height and width based on given depth
get_max_width_from_depth  = lambda depth: int(2 * depth)
get_max_height_from_depth = lambda depth: int(1. * depth)
# def get_max_width_from_depth(depth):
#     return int(2 * depth) # approximate 4 * depth
# def get_max_height_from_depth(depth):
#     return int(1. * depth) #approximate int(1.25 * depth)

def get_directed_neighbors(viewport_coordinates, depth, direction=None, voxel_size=VOXEL_SIZE, vision_limits=False):
    '''Get world voxel centers within the `viewfield` from the `origin viewport` 
    towards a given `scipy rotation` (`direction`) and maximum `depth`.'''
    if direction is None:
        direction = get_rotation_angles(random_direction=True)
    
    #to be modified according to the new estimations:
    lookat_offset = np.array([0, 0, depth]) * VOXEL_SIZE
    view_width    = get_max_width_from_depth(depth)   #max_x 
    view_height   = get_max_height_from_depth(depth)  #max_y
    view_depth    = depth                             #max_z

    # view_width    = 4 * depth         #max_x 
    # view_height   = int(1.25 * depth) #max_y
    # view_depth    = depth             #max_z
    # view_width    = int(2 * depth)    #max_x 
    # view_height   = int(1. * depth)   #max_y
    # view_depth    = depth             #max_z

    directed_screen   = get_directed_view_neighbors_indexes(view_width, view_height, view_depth, vision_limits=vision_limits)
    #neighbor_centers  = mock_point + directed_screen * voxel_size - .5 * voxel_size

    rotated_neighbors = viewport_coordinates + get_rotated_coordinates(directed_screen * voxel_size - .5 * voxel_size\
                                                            , scipy_model_rotation=direction)
    return rotated_neighbors


def get_directed_view_neighbors_indexes(max_x, max_y, max_z, vision_limits=False, direction_type="180"):
    '''Get integer indexes for a viewpoint centered in (0,0,0).
    Output shape should be (max_x, max_y, max_z, 3).
    Coordinates should be an int np array with: 
    [-x/2...x/2, -y/2...y/2, 1..z or only z (based vision_limits parameter)]'''
    
    if vision_limits:
        # vision_limits_plane is only the limit vision screen:
        vision_limits_plane = np.rollaxis(np.mgrid[  
              -max_x//2 : max_x//2\
            , -max_y//2 : max_y//2\
            , max_z-1   : max_z], 0, 4).reshape((-1, 3))
        directed_neighbors = vision_limits_plane
    else:
        full_view_field = np.rollaxis(np.mgrid[ 
              -max_x//2 : max_x//2\
            , -max_y//2 : max_y//2\
            , 0         : max_z], 0, 4).reshape((-1, 3))
        directed_neighbors = full_view_field
    return (directed_neighbors + 1) * np.array([1, 1, -1])

