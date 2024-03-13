# generated using gpt prompt:
# _I have set of  more than 3 vertices stored in a numpy array vertices_np. Give me a function to generate points inside the polygon determined by the vertices._

import numpy as np
import open3d as o3d
from utils.scripts.interest_heuristic_0 import get_o3d_pcd_from_coordinates


def is_point_inside_polygon(point, vertices):
    x, y = point
    winding_number = 0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        if y1 <= y:
            if y2 > y and (x - x1) * (y2 - y1) > (y - y1) * (x2 - x1):
                winding_number += 1
        elif y2 <= y and (x - x1) * (y2 - y1) < (y - y1) * (x2 - x1):
            winding_number -= 1
    return winding_number != 0

def generate_points_inside_polygon(vertices, num_points):
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    
    points = []
    while len(points) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = (x, y)
        if is_point_inside_polygon(point, vertices):
            points.append(point)
    return np.array(points)

def get_dense_o3d_block_from_bounds(loc_boudns, num_dense_points=1000):
    minX, minY, minZ = loc_boudns.min(axis=0)
    maxX, maxY, maxZ = loc_boudns.max(axis=0)

    print("Minimum bounds x,y,z: ", minX, minY, minZ)

    
    hhcolors = np.repeat(np.array([217,  55, 110]), num_dense_points).reshape((3, -1)).T / 255

    dense_block_np = np.vstack([minX + (maxX - minX) * np.random.random(num_dense_points)\
    , minY + (3*maxY - minY) * np.random.random(num_dense_points)\
    , minZ + (maxZ - minZ) * np.random.random(num_dense_points)]).T


    dense_block_pcd = get_o3d_pcd_from_coordinates(dense_block_np, hhcolors)

    # loc_bounds_rolled = np.roll(loc_boudns, 4, axis=0)
    loc_bounds_rolled = loc_boudns[[1,0,2,3]] # Replace with appropriate vertex permutation to get non-intersecting polygon

    vertices_np = np.vstack([loc_bounds_rolled[:,0], loc_bounds_rolled[:,2]]).T#np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    num_points            = num_dense_points#10
    points_inside_polygon = generate_points_inside_polygon(vertices_np, num_points)

    points_inside_polygon_y = minY + (3*maxY - minY) * np.random.random(num_points)

    points_inside_polygon_xyz = np.vstack([points_inside_polygon[:,0]\
                                           , points_inside_polygon_y\
                                           , points_inside_polygon[:,1]]).T
    #print("Points inside the polygon:")
    #print(points_inside_polygon_xyz)

    dense_block_pcd = get_o3d_pcd_from_coordinates(points_inside_polygon_xyz, hhcolors)
    
    return dense_block_pcd, points_inside_polygon_xyz



from scipy.spatial.transform import Rotation as R
from directed_utility_inputs_4 import get_rotated_up_front_lookat

def initialize_o3d_in_location(pd_endry, geometries, intialization=None):

    x, y, z    = pd_endry[["x", "y", "z"]]

    xr, yr, zr = pd_endry[["xh", "yh", "zh"]]

    scipy_model_rotation = R.from_euler("xyz", [xr, yr, 180-zr], degrees=True)
    r_front, r_lookat, r_up = get_rotated_up_front_lookat([x, y, z], scipy_model_rotation, lookat=np.array([0,0,16]))

    #     vis = o3d.visualization.VisualizerWithEditing()# Does not support more than one geometry
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for pcd in geometries:
        vis.add_geometry(pcd)


    ctr = vis.get_view_control()
    if intialization == "ufl":
        ctr.set_up(r_up)  # set the positive direction of the x-axis as the up direction
        ctr.set_front(r_front)  # set the positive direction of the x-axis toward you
        ctr.set_lookat(r_lookat)  # set the original point as the center point of the window
        ctr.set_zoom(.15)
    else:
        #extrinsic = extrinsic_matrix(x, y, z, xr, yr, zr)
        extrinsic         = np.eye(4)
        extrinsic[:3, :3] = scipy_model_rotation.as_matrix()
        extrinsic[:3, 3]  = [x, y, z]
    
        #This line will obtain the default camera parameters .
        camera_params = ctr.convert_to_pinhole_camera_parameters() 
        camera_params.extrinsic = np.linalg.inv(extrinsic)#extrinsic.T# your desired matrix
        #     ctr.set_zoom(1)
        #     ctr.set_up(np.array([0,0,1]))
        # leaving camera intrinsics untouched
        ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.run()
    vis.destroy_window()