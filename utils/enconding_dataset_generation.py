import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import time
np.random.seed(1)

# from Interest_Inference_Heuristic.ipynb
from utils.scripts.helper_ply_SensatUrban_0  import DataProcessing
from interest_heuristic_0              import color_point_cloud_with_binary_interst

from utils.scripts.directed_utility_inputs_4 import get_random_numerical_xyz_angles, get_rotation_angles\
                                            , get_rotated_up_front_lookat

from utils.scripts.o3d_interactive_visualization import load_view_point#, load_multiple_view_points

from utils.scripts.helper_ply_SensatUrban_0 import sem_color_to_name, ins_dict, Plot
import os
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
# from tqdm import tqdm
tqdm.pandas()

def generate_visual_dataset(scene_path, num_locations, dirs_per_loc, color_dictionary, visual, rgb="semantic", width=256, height=256): 
    '''
    Main method of generating data frame table and folder structures based on input configuration
    '''
    # Semantic colors point cloud:
    fl_annotated = DataProcessing().read_ply_data(scene_path)#, with_interest_label=True)
    
    if rgb == "semantic":
        pcd = Plot().draw_pc_sem_ins(pc_xyz=fl_annotated[0], pc_sem_ins=fl_annotated[2], return_pc=True)
    else:
        pcd = Plot().draw_pc(pc_xyzrgb=np.hstack([fl_annotated[0], fl_annotated[1]]), return_pc=True)
    
    vg  = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=.25)

    o3d_scene    = vg#_rgb# vg_semantic #read semantic
    #o3d_scene    = vg_semantic #read semantic

    data_store = f".{scene_path.strip('.ply')}_{rgb}_locs-{num_locations}_dirs-{dirs_per_loc}{'_visual' if visual else ''}"
    #0. create storage folder 
    if not os.path.exists(data_store):
        os.makedirs(data_store)
        print(f"Created data storage at:\n\t{data_store}")
    #1. Generate values and images
    xyz_xyz_hats                  = sample_xyz_xyz_hat_from_scene(o3d_scene\
                                                                  , num_locations=num_locations\
                                                                  , offset=np.array([5, 5, -1])\
                                                                  , with_repetitions=dirs_per_loc\
                                                                  , discrete_factor=5)
    visibility_dataset_df         = pd.DataFrame(xyz_xyz_hats)
    visibility_dataset_df.columns = ["x", "y", "z", "xh", "yh", "zh"]
    #2. create pandas and looping every entry:
        #a. save image and 
        #b. return scalar distribution for pd column
    def pd_looping_image_and_distribution_generation(pd_row):
        xyz_xyz_hat=[pd_row[i] for i in range(6)]
        #1. generate labels and image  
        label_dist = get_scalar_field_value_sematic(xyz_xyz_hat, o3d_scene\
                                             , color_dictionary=color_dictionary, return_visual=visual, width=width, height=height)
        #if real world input and semantic labels outputs, one neeeds to take two screenshots.
        #get_visual_from_xyz_xyz_hat(xyz_xyz_hat, pcd_scene=pcd_color)
        
        if visual:
            #2. save image
            label_dist, rgb_img = label_dist
            image_path = data_store + f"/{'_'.join(np.array(xyz_xyz_hat).astype(str))}.jpeg"
            im         = Image.fromarray(rgb_img)
            #im.save(image_path, 'TIFF')
            image_name = str(pd_row.name) + ".jpg"
            im.save(data_store + "/" + image_name, 'TIFF')
        #3. return label_distribution
        pd_row["f_xyz"]      = label_dist.astype(int)
        pd_row["image_name"] = image_name
        return pd_row#label_dist.astype(int), image_name

    #3. 
    #Generate outputs for each respective input:
    #visibility_dataset_df[["f_xyz", "image_name"]]      = 
    visibility_dataset_df = visibility_dataset_df.progress_apply(pd_looping_image_and_distribution_generation, axis=1)
#     visibility_dataset_df["image_name"] =  
    visibility_dataset_df.to_csv(data_store + "/locations.csv", index=False)
    classes_df = pd.DataFrame({"color":ins_dict.keys(), "class":ins_dict.values()})
    classes_df.to_csv(f"{data_store}/classes_index.csv", index=False)
    return visibility_dataset_df, classes_df


# np.random.seed(1)
# generate_visual_dataset(scene_path, 5, 2, color_dictionary=ins_dict, visual=True, rgb='semantic')
# np.random.seed(1)
# generate_visual_dataset(scene_path, 5, 2, color_dictionary=ins_dict, visual=True, rgb='real-color')
# np.random.seed(1)
# generate_visual_dataset(scene_path, 4, 2, color_dictionary=ins_dict, visual=False)
# np.random.seed(1)
# generate_visual_dataset(scene_path, NUM_LOCATIONS, DIRS_PER_LOC, ins_dict, VISUAL, rgb='semantic')



def sample_xyz_from_scene(pcd_scene, num_locations=1, offset=np.array([5, 5, 0])):
    """Return a random uniform xyz location within the pcd_scene plus minus the given offset"""
    
    min_range    = pcd_scene.get_min_bound() - offset
    max_range    = pcd_scene.get_max_bound() + offset
    max_distance = max_range - min_range
    
    xyz = min_range + np.random.random((num_locations, 3)) * max_distance
    
    return xyz

def sample_xyz_xyz_hat_from_scene(pcd_scene, num_locations=1, offset=np.array([5, 5, 0])\
                                  , with_repetitions=1, discrete_factor=5):
    """Return a random uniform xyz_xyz_hat location and angels within the pcd_scene plus minus the given offset
    xyz_xyz_hat.shape = (N, 6)
    xyz_hat: [0, 360) with some visual constraints described in get_random_numerical_xyz_angles
    """

    if with_repetitions > 1: # for each location draw randomly with_repetitions angles.
        xyz_locations = sample_xyz_from_scene(pcd_scene, num_locations, offset)
        
        #Repeat each location multiple with_repetitions:
        xyz_locations = np.repeat(xyz_locations, with_repetitions, axis=0)
        xyz_angles    = get_random_numerical_xyz_angles(num_locations * with_repetitions)
        
        
        #Discretize locations and angles according according to discrete_factor
        xyz_locations = (xyz_locations//discrete_factor)*discrete_factor
        xyz_angles    = (xyz_angles//discrete_factor)*discrete_factor
        
        xyz_xyz_hat   = np.hstack([xyz_locations, xyz_angles])
      
    if with_repetitions <= 1:
        xyz_locations = sample_xyz_from_scene(pcd_scene, num_locations, offset)
        xyz_angles    = get_random_numerical_xyz_angles(num_locations)
        xyz_xyz_hat   = np.hstack([xyz_locations, xyz_angles])
    
    return xyz_xyz_hat.astype(int) # In any case discretize angles and locations at least to be ints.

def get_visual_from_xyz_xyz_hat(xyz_xyz_hat, pcd_scene, zoom=.15, width=256, height=256):#.15):
    '''
    xyz_xyz_hat:      location and angles x, y, z, \hat x, \hat y, \hat z
    pcd_scene:        scene, either o3d point cloud or voxel_grid

    returns np_rgb_img (W, H, 3) 0..255 values
    '''
    
    xyz                     = xyz_xyz_hat[:3]
    xyz_hat                 = xyz_xyz_hat[3:]
    
    #Get visual information from the scene
    scipy_rotation          = get_rotation_angles(custom_direction=xyz_hat)
    r_up, r_front, r_lookat = get_rotated_up_front_lookat(origin=xyz, scipy_model_rotation=scipy_rotation)
    
    rgb_img                 = load_view_point(pcd_scene, flu=(r_front, r_lookat, r_up), zoom=zoom, width=width, height=height)
    return rgb_img




def normalize_num_pixels(num_interest_pixels, num_total_pixels, normalization=np.log):
    '''For raw percentage: pass normalization as lambda x: x'''
    return normalization(num_interest_pixels) / normalization(num_total_pixels)


#Deprecated - not using zooms anymore also no need of get_picked_2D_screen_location
def get_scalar_field_value(xyz_xyz_hat, pcd_scene\
                           , interest_color=np.array([217,  55, 110]), return_visual=False\
                          , pixel_normalization_rate=np.sqrt, zooms=[.15]\
                         , return_raw_values=False):
    '''
    #Deprecated - not using zooms anymore also no need of get_picked_2D_screen_location
    #Replaced with get_scalar_field_value_sematic
    xyz_xyz_hat:      location and angles x, y, z, \hat x, \hat y, \hat z
    pcd_scene:        scene, either o3d point cloud or voxel_grid
    interest_color:   if is np array color than return number of pixels of specified color
        if None return distribution of colors over 1000 pixels.
    scalar_variety:   Default "car percentage", i.e. counting number of pixels with color hhinterest
    return_raw_values: Ignore normalization of the number of pixels: True or False

    returns: f_xyz_xyz_hat, rgb_imgs - scalar field value f_xyz_xyz_hat (and visual rgb_img if return_visual is True)
    '''
    f_xyz_xyz_hat = [] 
    rgb_imgs      = []
    #for zoom in zooms[::-1]:
    for zoom in zooms:
        rgb_img                 = get_visual_from_xyz_xyz_hat(xyz_xyz_hat, pcd_scene, zoom)
        rgb_imgs.append(rgb_img)

        #Get scalar value based on visual inforamtion (count pixels)
        interest_pixels         = get_picked_2D_screen_location(rgb_img, interest_color, all_ids=True)
        num_interest_pixels     = 0 if np.any(np.isnan(interest_pixels[0])) else len(interest_pixels[0]) 
        num_total_pixels        = np.prod(rgb_img.shape[:2])

        #f_xyz_xyz_hat           = np.log(num_interest_pixels+1) / np.log(num_total_pixels+1)
        #f_xyz_xyz_hat           = num_interest_pixels / num_total_pixels
#         f_xyz_xyz_hat.append(normalize_num_pixels(num_interest_pixels+1, num_total_pixels+1, pixel_normalization))
        if return_raw_values:
            f_xyz_xyz_hat.append(num_interest_pixels)
        else:
            f_xyz_xyz_hat.append(normalize_num_pixels(num_interest_pixels, num_total_pixels, pixel_normalization_rate))
    
    if return_raw_values:
        f_xyz_xyz_hat.append(num_total_pixels)
    if return_visual:
        return f_xyz_xyz_hat, rgb_imgs

    return f_xyz_xyz_hat

def get_scalar_field_value_sematic(xyz_xyz_hat, pcd_scene \
                           , color_dictionary=None, return_visual=True, zoom=.15, width=256, height=256):
    '''
    xyz_xyz_hat - location (6, 1), pcd_scene - o3d geometry
    color_dictionary - colors to be looking for;
    return_visual - return screenshot or not.
    
    returns: label_dist, rgb_img
    '''
    #1. Get screenshot
    rgb_img                 = get_visual_from_xyz_xyz_hat(xyz_xyz_hat, pcd_scene, zoom=zoom, width=width, height=height)#.15)
    #2. Count pixels of each independent color:
    labels, counts = np.unique(np.reshape(rgb_img, (-1, 3)), axis=0, return_counts=True)
    #3. Intersect all the colors with dictionary colors:
    valid_colors = np.apply_along_axis(lambda rgb: tuple(rgb) in color_dictionary.keys(), axis=1, arr=labels)
    label_ids    = valid_colors#np.logical_and(np.any(labels!=[255, 255, 255], axis=1), counts>1000)
    found_colors = labels[label_ids]

    #4. Index found colors in dictionary of colors:
    valid_label_colors = list(color_dictionary.keys())
    found_labels       = [valid_label_colors.index(tuple(l)) for l in found_colors]

    #5. Compute distribution of labels 
    label_dist               = np.zeros(len(valid_label_colors)+1)
    label_dist[found_labels] = counts[label_ids] #/ counts[label_ids].sum()
    #5.1. The last position will represent all the pixels not belonging to any class
    label_dist[-1] = counts.sum() - counts[label_ids].sum()
    if return_visual:
        return label_dist, rgb_img
    
    return label_dist

# get_scalar_field_value_sematic(xyz_xyz_hat, vg_semantic, color_dictionary=ins_dict)


# Deprecated - using ZOOMS approach.
def visualize_input_point_and_visual_information(xyz_xyz_hat, geometries\
                                                 , scoring_method=get_scalar_field_value\
                                                 , pixel_normalization_rate=np.sqrt\
                                                 , zooms=[.3, .15, .05]):
    """
    Deprectaed - using ZOOMS approach
    Visualize information about an input point inside geometries:
    scoring_method : f(input_xyz_and_angles_tuple, geometry) and hhcolor color of pixels to be counted
    if len geometries == 3
    than geometries   = pcd, pcd_interest, vg_interest
    hhcolor is harcoded interest - hhcolor = np.array([217,  55, 110])
    zooms: list of zooms to which rate to generate the visualizations
    """
    hhcolor = np.array([217,  55, 110])
    if len(geometries)==1: # Visualize the point in one geometry - 
        pcd = geometries[0]
        
        #rgb_real                            = get_visual_from_xyz_xyz_hat(xyz_xyz_hat, pcd)# If output doesn't need to be computed
        f_xyz_xyz_pcd_hat, rgb_pcd_interest = scoring_method(xyz_xyz_hat, pcd, hhcolor, return_visual=True, zooms=zooms)
        print(f"Zoom rates:                    {zooms}")
        print(f"Output sparse point cloud:     {f_xyz_xyz_pcd_hat}")
        fig, ax = plt.subplots(nrows=1, ncols=len(zooms), figsize=(20, 40))
        for i, zoom in enumerate(zooms):
            ax[i].imshow(rgb_pcd_interest[i]); ax[i].set_title(f"zoom: {zoom} - annotated interest: {f_xyz_xyz_pcd_hat[i]:.2f}")
        #plt.title()
        plt.show()
    
    if len(geometries)==3:#same point in multiple geometries - pcd, pcd_interest, vg_interest
        
        pcd, pcd_interest, vg_interest = geometries
        
        rgb_reals                            = [get_visual_from_xyz_xyz_hat(xyz_xyz_hat, pcd, zoom=z) for z in zooms]
        f_xyz_xyz_pcd_hat, rgb_pcd_interest = scoring_method(xyz_xyz_hat, pcd_interest, hhcolor\
                                                             , return_visual=True\
                                                             , pixel_normalization_rate=pixel_normalization_rate\
                                                            , zooms=zooms)
        f_xyz_xyz_vg_hat, rgb_vg_interest   = scoring_method(xyz_xyz_hat, vg_interest, hhcolor\
                                                             , return_visual=True\
                                                             , pixel_normalization_rate=pixel_normalization_rate\
                                                            , zooms=zooms)


        
        print(f"Input xyz and xyz_hat:         ({', '.join(xyz_xyz_hat.astype(int).astype(str))})")
        print(f"Zoom rates:                    {zooms}")
        print(f"Output sparse point cloud:     {np.array(f_xyz_xyz_pcd_hat).round(4)}")
        print(f"Output dense voxel grid cloud: {np.array(f_xyz_xyz_vg_hat).round(4)}")

        for i, zoom in enumerate(zooms):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 40))

            ax[0].imshow(rgb_reals[i]); ax[0].set_title("Point cloud visual raw RGB information, w/o annotated interest")
            ax[1].imshow(rgb_pcd_interest[i]); ax[1].set_title(f"Point cloud visual, at zoom {zooms[i]}, interest percentage of {f_xyz_xyz_pcd_hat[i]:.2f}")
            ax[2].imshow(rgb_vg_interest[i]); ax[2].set_title(f"Voxel Grid visual, at zoom {zooms[i]}, interest percentage of  {f_xyz_xyz_vg_hat[i]:.2f}")

            # plt.legend(True)
            plt.show()
            
# visualize_input_point_and_visual_information(xyz_xyz_hat[15], [pcd, pcd_interest, vg_interest]\
#                                              , pixel_normalization_rate=np.sqrt, zooms=[.8,.3,.15,.05,.001])

