
#ABANDONED - On the "Depth Prediction" train of thought
#Developed in 9_Point_Cloud_ScreenShot.ipynb

import matplotlib.pyplot as plt
import numpy  as np
import open3d as o3d


def get_rgbd_from_selection(pcd):
    '''
    from `pcd` point cloud navigate and select desired region:
    return depth, rgb_img, pcd_selected
    '''
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = np.array([[1,0,0,1],
                                               [0,1,0,0],
                                               [0,0,1,2],
                                               [0,0,0,1]])
    camera_parameters.intrinsic.set_intrinsics(width=1920, height=1080, fx=1000, fy=1000, cx=959.5, cy=539.5)

    viewer = o3d.visualization.VisualizerWithEditing()
    viewer.create_window()
    viewer.add_geometry(pcd)

    viewer.run()

    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(camera_parameters)


    depth        = np.array(viewer.capture_depth_float_buffer())
    rgb_img      = np.array(viewer.capture_screen_float_buffer())
    pcd_selected = viewer.get_cropped_geometry()
    
    return depth, rgb_img, pcd_selected, viewer



import numpy as np
import open3d as o3d
# def align_image_depth_prediction(prediction):
def get_3d_np_from_2d_depth(depth_prediction):
    '''
        given depth for each 2D location in image, create 3D locations
        depth_prediction -> img_3d_locations
    '''
    #Idea from:
    #https://stackoverflow.com/questions/73891858/
    print("creating 3D locations from depth prediction")
    x_max, y_max = depth_prediction.shape
    #xy_locations = np.mgrid[x_max:0:-1, y_max:0:-1]
    xy_locations = np.mgrid[x_max:0:-1, 0:y_max:1]

    img_3d_locations = np.stack([xy_locations[1], xy_locations[0], depth_prediction], axis=2)#.shape

    # img_3d_locations.shape, img.shape, np.vstack(img_3d_locations).shape
    return img_3d_locations

def get_o3d_pcd_from_image_and_locations(rgb_img, img_3d_locations):
    """
        Given RGB np image and 3d locatioans, return o3d pcd
        rgb_img, img_3d_locations -> pcd
    """
    print("creating Open3D point cloud from image and depth")
    
    pcd          = o3d.geometry.PointCloud()
    np_locations = np.vstack(img_3d_locations)
    np_rgb       = np.vstack(rgb_img)
    pcd.points   = o3d.utility.Vector3dVector(np_locations)
    pcd.colors   = o3d.utility.Vector3dVector(np_rgb)
    
    return pcd
