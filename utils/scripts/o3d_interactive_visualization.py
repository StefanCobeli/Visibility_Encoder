
# ADAPTED from OPEN3D python examples -- interactive_visualization
# examples/python/visualization/interactive_visualization.py

import numpy as np
from copy import copy
import open3d as o3d

#for load_view_point from xyz c00rdinates and angles
from scipy.spatial.transform import Rotation as R


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

    MAX_ANGLE         = 360#720 #540 #360 450 540
    rotation_angles = [-MAX_ANGLE] + [2 * MAX_ANGLE // num_cam_rotations for i in range(num_cam_rotations)]  + [-MAX_ANGLE]
    
    # Determining the Proper Camera Position:
    # https://github.com/isl-org/Open3D/issues/2338
    for i in range(len(rotation_angles)):
        # view_controler.set_lookat(lookat)
        view_controler.rotate(rotation_angles[i], rotation_angles[i])
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
    
    if len(camera_parameters) > 1:
        shots = load_multiple_view_points(pcd, camera_parameters, filename=None, visualize=False)
        return shots
    
    for i in range(len(camera_parameters)):
        np_rgb_img = load_view_point(pcd, camera_parameters[i])
        shots.append(np_rgb_img)
        
    return shots


def load_multiple_view_points(pcd, camera_parameters, filename=None, visualize=False):
    """same as load_view_point, but don't create_window for each camera param 
    #TO DO - OPTIMIZATION - add method parameters to change camera intirinsics 
    with param.intrinsic.set_intrinsics
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    if filename:
        param = o3d.io.read_pinhole_camera_parameters(filename)
        camera_parameters = [param]
    vis.add_geometry(pcd)
    
    rgb_image_shots = []
    for param in camera_parameters:
        ctr.convert_from_pinhole_camera_parameters(param)
        if visualize:
            vis.run()
        rgb_img = vis.capture_screen_float_buffer(do_render=True)
        rgb_image_shots.append((255.0 * np.asarray(rgb_img)).astype(np.uint8))
    vis.destroy_window()
    return rgb_image_shots


def load_view_point(pcd, param=None, filename=None, visualize=False, return_vc=False, xyz_ca=None, extrinsic=None, flu=None, zoom=.15, width=256, height=256):
    """given point cloud and param = o3d.io.read_pinhole_camera_parameters(filename)
    extrinsic:  4x4 extrinsic matrix;
    flu:        three tuple with (front, lookat, up) can take place of intrisic camera parameters
    xyz_ca:     xyz coordinates and angels - list with 6 elements x,y,z cooridinates and x,y,z angle rotations;
    return_vc:  returns also view controler so camera parameters can also be retrieved.
    #TO DO - OPTIMIZATION - add method parameters to change camera intirinsics 
    with param.intrinsic.set_intrinsics
    """
    vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1024, height=1024)
    vis.create_window(width=width, height=height)
    ctr = vis.get_view_control()
    if filename:
        param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    if flu is not None:
        r_front, r_lookat, r_up = flu
        ctr.set_up(r_up)  # set the positive direction of the x-axis as the up direction
        ctr.set_front(r_front)  # set the positive direction of the x-axis toward you
        ctr.set_lookat(r_lookat)  # set the original point as the center point of the window
        ctr.set_zoom(zoom)

    if xyz_ca is not None: #and extrinsic is None
        rm = R.from_euler("xyz", xyz_ca[3:], degrees=True).as_matrix()
        # and translation to obtain extrinsic matrix
        extrinsic         = np.eye(4)
        extrinsic[:3, :3] = rm
        extrinsic[:3, 3]  = xyz_ca[:3]

    if extrinsic is not None:
        #This line will obtain the default camera parameters .
        camera_params = ctr.convert_to_pinhole_camera_parameters() 
        print("Prev cam:\n\t", camera_params.extrinsic)
        #camera_params.extrinsic = np.linalg.inv(extrinsic)
        camera_params.extrinsic = extrinsic
        param = camera_params#extrinsic
        print("New cam:\n\t", param.extrinsic)
        #ctr.convert_from_pinhole_camera_parameters(param)

    if param is not None:
        ctr.convert_from_pinhole_camera_parameters(param)
    if visualize:
        vis.run()
    rgb_img = vis.capture_screen_float_buffer(do_render=True)
    # print(rgb_img)
    vis.destroy_window()
    if return_vc:
        return ctr, (255.0 * np.asarray(rgb_img)).astype(np.uint8)
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



def load_view_controler(filename, lookat=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)
    if lookat:
        ctr.set_lookat(lookat)
    vis.destroy_window()
    return ctr




from collections import namedtuple

def load_view_controler_and_picked_points(location_saved_id, pcd, storage_path="./np_files/"):
    '''
    return view_controler, picked_points
    '''

    PickedPointLoaded = namedtuple(typename="PickedPointLoaded", field_names=["index", "coord"])
    #Load view control and picked points:
    #location_saved_id = 1

    #1. View points 
    #b. and load
    picked_points_ids_np      = np.load(f"./{storage_path}/picked_points_ids_{location_saved_id}.npy")
    # print([pcd.points[ppin] for ppin in picked_points_ids_np])
    # print(np.mean([pcd.points[ppin] for ppin in picked_points_ids_np], axis=0))
    loaded_picked_points_list = [PickedPointLoaded(ppin, pcd.points[ppin]) \
                                 for ppin in picked_points_ids_np]

    # 2. View Controler
    # b. Load
    centroid_lookat = np.mean([pcd.points[ppin] for ppin in picked_points_ids_np], axis=0)
    loaded_view_controler = load_view_controler(f"./{storage_path}/picked_centroid_{location_saved_id}.json"\
                                               , lookat=None)

    view_controler, picked_points = loaded_view_controler, loaded_picked_points_list
    
    return view_controler, picked_points


def save_view_controler_and_picked_points(view_controler, picked_points, location_saved_id, storage_path="./np_files/"):
    '''
    return True / False
    '''
    # PickedPointLoaded = namedtuple(typename="PickedPointLoaded", field_names=["index", "coord"])
    #Save view control and picked points:
    #location_saved_id = 1

    #1. View points 
    #a. Save 
    np.save(f"./np_files/picked_points_ids_{location_saved_id}.npy", [pp.index for pp in picked_points])
    
    # 2. View Controler
    #a. Save
    centroid_cam_params   = view_controler.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(f"./np_files/picked_centroid_{location_saved_id}.json"\
                                           , centroid_cam_params)
    return True






if __name__ == "__main__":
    pcd_colored = o3d.io.read_point_cloud("../datasets/SensatUrban_Dataset/ply/train/cambridge_block_4.ply")
    points_1 = pick_points(pcd_colored)
    points_2 = pick_points(pcd_colored)

    print(points_1)
    print()
    print(points_2)
