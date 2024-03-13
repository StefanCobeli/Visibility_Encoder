
# ADAPTED from OPEN3D python examples -- interactive_visualization
# examples/python/visualization/interactive_visualization.py

import numpy as np
from copy import copy
import open3d as o3d


def pick_segmentation_anchor_points(pcd):
    '''
        return picked points from point cloud and view controler
    '''
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    view_controler = (vis.get_view_control())
    #param          = view_controler.convert_to_pinhole_camera_parameters()
    picked_points  = (vis.get_picked_points())
    vis.destroy_window()
    return view_controler, picked_points

def get_camera_parameters_from_controler(view_controler, lookat, num_cam_rotations=4, zoom=.05):
    '''
    Retun camera parameters list by rotating view_controler and looking at lookat.
    return camera_parameters
    '''
    camera_parameters = []

    MAX_ANGLE         = 720 #540 #360 450 540
    rotation_angles = [-MAX_ANGLE] + [2 * MAX_ANGLE // num_cam_rotations for i in range(num_cam_rotations)]  + [-MAX_ANGLE]
    
    # Determining the Proper Camera Position:
    # https://github.com/isl-org/Open3D/issues/2338
    for i in range(len(rotation_angles)):
        view_controler.set_lookat(lookat)
        view_controler.rotate(rotation_angles[i], 0)
        #view_controler.camera_local_rotate(rotation_angles[i], 0)
        #view_controler.set_zoom(zoom) #Better to NOT change zoom - Should dynamically depend on the size of the slected object
        camera_parameters.append(copy(view_controler.convert_to_pinhole_camera_parameters()))
        
    return camera_parameters

def get_screen_caputures_from_parameters(pcd, camera_parameters):
    '''
    given point cloud and camera parameters 
    return numpy rgb screenshot of point cloud from camera parameters
        # TO DO - OPTIMIZATION - change camera intirinsics with param.intrinsic.set_intrinsics
        NOTE: Does NOT work on MAC - load_view_point / convert_from_pinhole_camera_parameters 
        https://github.com/isl-org/Open3D/issues/897
    '''
    shots = []
    for i in range(len(camera_parameters)):
        np_rgb_img = load_view_point(pcd, camera_parameters[i])
        shots.append(np_rgb_img)
        
    return shots



def load_view_point(pcd, param, filename=None, visualize=False):
    """given point cloud and param = o3d.io.read_pinhole_camera_parameters(filename)
    #TO DO - OPTIMIZATION - add method parameters to change camera intirinsics 
    with param.intrinsic.set_intrinsics
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    if filename:
        param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    if visualize:
        vis.run()
    rgb_img = vis.capture_screen_float_buffer(do_render=True)
    # print(rgb_img)
    vis.destroy_window()
    # print(rgb_img)
    #return np.array(rgb_img)
    return (255.0 * np.asarray(rgb_img)).astype(np.uint8)


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()




if __name__ == "__main__":
    pcd_colored = o3d.io.read_point_cloud("../datasets/SensatUrban_Dataset/ply/train/cambridge_block_4.ply")
    points_1 = pick_points(pcd_colored)
    points_2 = pick_points(pcd_colored)

    print(points_1)
    print()
    print(points_2)
