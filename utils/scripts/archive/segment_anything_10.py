#Code adapted from predictor_example.ipynb

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# sam_checkpoint = "sa_models/sam_vit_h_4b8939.pth"
# # sam_checkpoint = "sa_models/sam_vit_l_0b3195.pth"
# # sam_checkpoint = "sa_models/sam_vit_b_01ec64.pth"
# device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_type = "default"


import open3d as o3d

def get_picked_2D_screen_location(reg_id_screenshot, selected_id, all_ids=False):
    '''
    Returns pixel coordinates from reg_id_screenshot, of color selected_id
    reg_id_screenshot: RGB image with pixels in [0..255]
    selected_id:       3-tuple RGB color  - [0..255]^3
    all_ids:           if True returns all pixels of color, if False returns median

    Expects colors in [0..255]^3
    Given rgb id image and color id of selected point, return 
    picked_x, picked_y 
    as median point with color selected_id.
    '''
    # https://stackoverflow.com/a/71471432
    # create mask for blue color in hsv
    # blue is 240 in range 0 to 360, so for opencv it would be 120
    #selected_id = np.array([239, 225, 247])

    lower              = selected_id - 1#- .25 
    upper              = selected_id + 1#+ .25

    mask               = cv2.inRange(reg_id_screenshot, lower, upper)
    if mask.sum() == 0:
        return np.nan, np.nan

    picked_y, picked_x = np.nonzero(mask)

    if not(all_ids):
        # picked_x, picked_y = np.median(picked_x).astype(np.uint8), np.median(picked_y).astype(np.uint8)
        picked_x, picked_y = np.median(picked_x), np.median(picked_y)
    
    return picked_x, picked_y

def get_2D_screen_location_of_picked_points(shot_w_id, pcd_ids, picked_points):
    '''Return list of selected points using get_picked_2D_screen_location
    Given 2D image with points as unique colors, point cloud and picked point return picked points
    shot_w_id        - 2D image with rgb pixels [0..255]
    pcd_ids          - O3D point cloud with 0..1 colored pixels
    picked_points    - list of O3D points with index and color
    picked_points_2D - list of (x, y) tuples in 2D screen
    '''
    picked_points_2D = []
    for pp in picked_points:
        selected_id      = (255.0 * np.asarray(pcd_ids.colors[pp.index])).astype(np.uint8)

        picked_x, picked_y = get_picked_2D_screen_location(shot_w_id, selected_id)
        #print(picked_x, picked_y)
        #if np.isnan(picked_x):# is np.nan:
        #    continue

        picked_points_2D.append([picked_x, picked_y])
    
    return picked_points_2D

def prepare_point_cloud_segmentation(pcd_colored):
    '''
        Given o3d point cloud `pcd_colored`;
        Return: same point cloud but with ids instead of colors
    '''
    positions    = np.array(pcd_colored.points)
    point_colors = np.array(pcd_colored.colors)

    num_points          = positions.shape[0]
    unique_point_ids    = generate_color_ids_array(num_points)
    
    pcd_ids             = o3d.geometry.PointCloud()
    pcd_ids.points      = o3d.utility.Vector3dVector(positions)     
    pcd_ids.colors      = o3d.utility.Vector3dVector(unique_point_ids / 255)
    
    unique_point_ids_1D = get_ids_1D_from_ids_2D(unique_point_ids)

    return pcd_ids, point_colors, unique_point_ids, unique_point_ids_1D

# print("Test if point indexes in pcd_id match indexes in colored pcd:")
# print("\t", np.all(np.equal(
#                     [pcd_ids.points[pid] for pid in [pp.index for pp in picked_points]] \
#                     , [pcd.points[pid] for pid in [pp.index for pp in picked_points]]))
#      )


import sys
sys.path.append("..")
sys.path.append("../utils/github_utils/segment-anything")
# from utils.github_utils.segment_anything import sam_model_registry, SamPredictor
from segment_anything import sam_model_registry, SamPredictor

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)

def set_up_segment_anything(path_to_sa="./utils/github_utils/segment-anything/"\
                            , path_to_sa_model="sa_models/sam_vit_h_4b8939.pth"):
    '''Rturn Segment Anything predictor'''
    sys.path.append(path_to_sa)
    model_type = "default"
    sam        = sam_model_registry[model_type](checkpoint=path_to_sa+path_to_sa_model)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device=device)
    predictor = SamPredictor(sam)
                            
    return predictor

path_to_sa       = "./utils/github_utils/segment-anything/"
path_to_sa_model = "sa_models/sam_vit_h_4b8939.pth" #"sam_vit_b_01ec64.pth", "sam_vit_h_4b8939.pth", "sam_vit_l_0b3195.pth"
# predictor        = set_up_segment_anything(path_to_sa, path_to_sa_model)


def generate_color_ids_array(num_points, random=True):
    ''''Return unique point ids as RGB colors [0, 255]^3'''
    potential_colors = np.rollaxis(np.mgrid[  0:256, 0:256, 0:256], 0, 4).reshape((-1, 3))
    if random: 
        np.random.seed(1)
        n_total         = potential_colors.shape[0]
        random_ids      = np.random.choice(np.arange(n_total), num_points, replace=False)
        selected_points = potential_colors[random_ids]
        pass
    else:
        selected_points = potential_colors[:num_points]
    return selected_points


def get_sam_masks_around_point(input_point, image, predictor):
    '''
    return masks, scores, logits
    '''
    predictor.set_image(image)
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )   
    return masks, scores, logits


def get_sam_masks_around_multiple_points(input_points, image, predictor):
    '''
    see Specifying a specific object with additional points
    return masks, scores, logits
    '''
    predictor.set_image(image)
    input_labels =  np.ones(len(input_points))
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )   
    return masks, scores, logits


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# Code developed in 10_Matching_RGBD_with_PCD.ipynb
import numpy as np
def replace_mask_with_color(image, mask, color=np.array([30, 144, 255])):
    '''Given image and mask, replace mask in image with color'''
    segmented_image       = image.copy()
    segmented_image[mask] = color
    return segmented_image



def get_ids_1D_from_ids_2D(ids_2D):
    '''Transform 3D colors into 1D string id'''
    np_row_to_3_string = lambda row: len(row) * "%s" % tuple(['0' * (3 - len(str(x))) + str(x) for x in tuple(row)])
    ids_1D = np.apply_along_axis(np_row_to_3_string, axis=1, arr=ids_2D)
    return ids_1D

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
    xy_locations = np.mgrid[x_max:0:-1, 0:y_max:1]//100
    img_3d_locations = np.stack([xy_locations[1], xy_locations[0], depth_prediction], axis=2)#.shape
    # img_3d_locations.shape, img.shape, np.vstack(img_3d_locations).shape
    return img_3d_locations


def plot_image(image, input_point=None, input_label=None, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if not(input_point is None):
        show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()
    