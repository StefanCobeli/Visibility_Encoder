import warnings; warnings.simplefilter('ignore') #Disable torch warnings
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# from lang_sam        import LangSAM
from PIL             import Image
from tqdm.notebook   import tqdm
# from skimage.measure import block_reduce
from utils.scripts.helper_ply_SensatUrban_0 import DataProcessing, write_ply


#SensatUrban categories:
# ins_colors = [[85, 107, 47],  # ground -> OliveDrab 0
#     [0, 255, 0],  # tree -> Green 1
#     [255, 165, 0],  # building -> orange 2 
#     [41, 49, 101],  # Walls ->  darkblue 3 
#     [0, 0, 0],  # Bridge -> black 4
#     [0, 0, 255],  # parking -> blue 5 
#     [255, 0, 255],  # rail -> Magenta 6
#     [200, 200, 200],  # traffic Roads ->  grey 7
#     [89, 47, 95],  # Street Furniture  ->  DimGray 8 
#     [255, 0, 0],  # cars -> red 9 
#     [255, 255, 0],  # Footpath  ->  deeppink 10
#     [0, 255, 255],  # bikes -> cyan 11
#     [0, 191, 255]  # water ->  skyblue 12
# ]
# ins_colors = [[round(r/257, 2), round(g/257, 2), round(b/257, 2)] for (r, g, b) in ins_colors]

# ins_names  = ["ground",            "tree",    "building", "Walls"
#             , "Bridge",            "parking", "rail",     "traffic Roads"
#             , "Street Furniture ", "cars",    "Footpath", "bikes"
#             ,  "water"
# ]

def annotate_ply_with_sam_interest(ply_path, text_prompt, pix_res_patch, meter_res_patch, overlap_factor, verbose=True):
    '''
    Generate interest annotated point cloud for ply_path.
    return fl_annotated - (xyz, rgb, label, interest)
    '''
    max_interest     = (pix_res_patch // meter_res_patch)**2
    fl               = DataProcessing().read_ply_data(ply_path)
    model            = LangSAM()

    interest_map, map_bounds = generate_interest_map(fl, sam_model=model\
                                 , text_prompt=text_prompt, verbose=verbose\
                                , dimp=meter_res_patch, overlap_factor=overlap_factor, resolution=pix_res_patch)

    interest_probability_map = interest_map / max_interest
    interest_array           = project_interest_map_to_points(fl[0], interest_probability_map, map_bounds)

    #pcd_interest             = color_point_cloud_with_binary_interst(fl, interest_array, random_color=True)


    interest_annotated_ply = np.hstack([fl[0], fl[1], fl[2].reshape(-1, 1), interest_array.reshape(-1, 1)])
    out_interest_path = f"./{ply_path.strip('.ply')}_sam_interest_{text_prompt}_annotated.ply"
    
    write_ply(filename=out_interest_path\
              , field_list=interest_annotated_ply\
              , field_names=['x', 'y', 'z', 'red', 'green',  'blue', 'class', "interest"])
    
    fl_annotated = fl[0], fl[1], fl[2], interest_array

    return fl_annotated



def generate_interest_map(fl, sam_model, text_prompt, verbose=False, dimp=16, overlap_factor=3, resolution=256):
    '''
    Return interest map meter by meter according to point cloud fl, sam model and text prompt:
    Subdivides point cloud in paches of dimp^2 meters overlaping by 1/overlap_factor and of resolution^2 pixels.
    
    fl - point cloud with cooridngates in fl[0] - numpy array (3, n) and fl[1] - np (3, n) colors [0,..,255]
    sam-model      - LangSAM() - https://github.com/luca-medeiros/lang-segment-anything/
    text_prompt    - description of objects of interest, e.g. "car"
    verbose        - If True print each segmented patch 
    dimp           - dimension of each patch in meters
    overlap_factor - 1 / how match do eatch 2 adjacent patches ovelap (e.g. 2 - ovelap a half)
    resolution     - resolution of each screenshot of a patch in pixels
    
    # max value in interest_map is (resolution//dimp)**2 if the entire patch was segmented
    return interest_map, map_bounds - x coordinates and y coordinates in real world for each interest_map cell.
    '''
    #world bounds
    min_wx, max_wx, min_wy, max_wy = fl[0][:, 0].min(), fl[0][:, 0].max(), fl[0][:, 1].min(), fl[0][:, 1].max()
    
    #Goal of interest approximation:
    #how many mask points are in each of the interest map locations.
    interest_map = np.zeros((int(max_wx - min_wx), int(max_wy - min_wy)))
    map_bounds   = np.arange(min_wx, max_wx+1), np.arange(min_wy, max_wy+1)

    #Patch starts and end coordinates in world space
    minxs = np.arange(min_wx, max_wx-dimp, dimp-dimp//overlap_factor)
    minys = np.arange(min_wy, max_wy-dimp, dimp-dimp//overlap_factor)

    #mins as a cartesian product of (x, y) mins
    patch_mins    = np.transpose([np.tile(minxs, len(minys)), np.repeat(minys, len(minxs))])
    patch_offsets = patch_mins - np.array([min_wx, min_wy])#fl[0].min(axis=0)[:2]
    
    for i, (minx, miny) in enumerate(tqdm(patch_mins)):
        bx, by = (minx, minx+dimp), (miny, miny+dimp)

        if verbose: print(f"Patch {i+1} / {len(patch_mins)}:")

        #Points inside patch correspondig to - minx, miny:
        xyz_filtered, rgb_filtered, ids_filtered = filter_xyz_rgb_to_bounds(fl[0], fl[1], bx, by)
        #Aerial image of patch correspondig to - minx, miny:
        image_pil, _, _ = get_rgb_patch_within_bounds(xyz=xyz_filtered, rgb=rgb_filtered, bins=(resolution, resolution))
        #segmented aerial image
        masks, boxes, phrases, logits = get_iterest_locations_lang_sam(image_pil, text_prompt=text_prompt, model=sam_model)
        
        if len(boxes) > 0:
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
            if verbose:
                print("Box estimative surfaces m^2: \n\t", [box_surface_estimation(b, dimp) for b in boxes])
                print("Prediction confiedences: \n\t", logits.unsqueeze(1))
                display_image_with_masks(image_pil, masks_np)
                display_image_with_boxes(image_pil, boxes, logits)    
        else:
            if verbose:
                plt.imshow(image_pil)
                plt.title(f"No {text_prompt} found in patch")
                plt.show()

        for j, mask in enumerate(masks):
            #If area of the car is greater than 20 m^2 skip
            if box_surface_estimation(boxes[j], dimp) > 20 and "car" in text_prompt:  #empirically it should be around ~ 10m^2
               continue
            interest_map = update_interest_map(interest_map, mask, patch_offsets[i], resolution//dimp)
        
        if verbose:
            plt.imshow(interest_map)
            plt.title(f"Updated interest map after Patch {i+1} / {len(patch_mins)}")
            plt.show()

            save_mask(interest_map, f"./inferred_interests/{i}_{len(patch_mins)}_interest_map_patch.png")

    # x coordinates and y coordinates in real world for each interest_map cell.
    # Each interst_map unit represents how many subpatches are of interest in each patch.
    # max value in interest_map is (resolution//dimp)**2 if the entire patch was segmented
    return interest_map, map_bounds #patch_mins
    #minxs, minys, np.transpose([np.tile(minxs, len(minys)), np.repeat(minys, len(minxs))]).shape




def project_interest_map_to_points(xyz, interest_probability_map, map_bounds):
    '''
    assign interest label probabilistically to each point;
    Given xyz np coordinates (n, 3), normalized meter by metere interest_map, delimited by map bounds
    return interest_array (n) \in {0, 1}.
    '''
    #pix_res_patch   = 128
    #meter_res_patch = 32

    #interest_array = np.zeros(fl[0].shape[0])
    #max_interest         = (pix_res_patch // meter_res_patch)**2
    interest_array       = np.zeros(xyz.shape[0])
    #interest_probability = interest_map / max_interest
    #for each point
    for i, point in tqdm(enumerate(xyz), total=xyz.shape[0]):
        #pass
        #interest_patches = np.argwhere(interest_map!=0)
        #get corresponding patch for point:
        pi, pj = (point[:2] - np.array([map_bounds[0][0], map_bounds[1][0]])).astype(int)


        if interest_probability_map[pi-1, pj-1]!=0:
            interest_chance   = interest_probability_map[pi, pj]
            interest_array[i] = np.random.choice(2, p=[1-interest_chance, interest_chance])

    return interest_array

def color_point_cloud_with_binary_interst(fl, interest_array, random_color=True, custom_color=[201,  84, 123]):
    '''
    return ord point cloud with rgb color if point is NOT of interest and label color if of interest.
    '''
    color = (np.concatenate([np.random.random(3)], axis=0)*255).astype(int) if random_color else np.array(custom_color)#pink

    interest_colors = np.zeros_like(fl[1])
    interest_colors[np.where(interest_array)]   = color
    interest_colors[np.where(1-interest_array)] = fl[1][np.where(1-interest_array)]

    pcd_interest = get_o3d_pcd_from_coordinates(fl[0], interest_colors/255)

    return pcd_interest

# o3d.visualization.draw_geometries([color_point_cloud_with_binary_interst(fl, interest_array, random_color=True)])




def filter_xyz_rgb_to_bounds(xyz, rgb, bx, by):
    '''
        cut xyz and rgb according to bx and by bounds in xyz.
        bx, by - bounds in world space as tuples (min, max)
        z coordinates are ignored.
        return xyz_filtered, rgb_filtered, filtered_ids
    '''
    
    ids_filtered = np.where((xyz[:,0] >= bx[0]) & (xyz[:,0] <=bx[1]) \
                            & (xyz[:,1] >= by[0]) & (xyz[:,1] <=by[1]))
    xyz_filtered = xyz[ids_filtered]
    if not(rgb is None):
        rgb_filtered = rgb[ids_filtered]
    else:
        rgb_filtered = None

    return xyz_filtered, rgb_filtered, ids_filtered


def get_rgb_patch_within_bounds(xyz, rgb, bins=(256, 256)):
    '''
        render xyz with rgb colors on resolution bins.
        bins acts as image resolution
        returns world located rendered np image 
        returns image_pil, xedges, yedges - image and real coodrdinates for each pixel.
    '''

    xyz_heatmap, xedges, yedges = np.histogram2d(xyz[:,0], xyz[:,1], bins=bins)
    rendered_channels           = []

    for i in range(3): #rgb heatmaps
        c_heatmap, _, _ = np.histogram2d(xyz[:,0], xyz[:,1], bins=bins, weights=rgb[:, i]) 
        #normalized_map  = (c_heatmap / xyz_heatmap).astype(int).reshape(-1,1)# * 255

        normalized_map = np.divide(c_heatmap, xyz_heatmap, out=np.ones_like(c_heatmap)*255\
                                   , where=xyz_heatmap!=0).reshape(-1,1)#*255#
        rendered_channels.append(normalized_map)

    rendered_np_patch = np.hstack(rendered_channels).reshape((*bins,3))

    image_pil = Image.fromarray(np.uint8(rendered_np_patch)).convert("RGB")

    return image_pil, xedges, yedges

def box_surface_estimation(bounds, patch_dim_meters, resolution=256):
    return ((bounds[2]-bounds[0]) * (bounds[3]-bounds[1]) / resolution**2) * patch_dim_meters**2

def get_iterest_locations_lang_sam(image_pil, text_prompt, model):
    """
        Apply language Segment Anything on image_pil with given text prompt
        image_pil - [0..256]^3
        return masks, boxes, phrases, logits
    """
    #image_pil = Image.open("./2D_Slice.png").convert("RGB")
    #text_prompt = "tree"#"automobile"#"parked car", "cars", "car"
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    return masks, boxes, phrases, logits


def update_interest_map(interest_map, mask, mask_offset, res_by_patch):
    '''
    Compute new map of interest based on found `mask` interest in a sub-patch with offset `mask_offset`
    interest_map - global map
    mask         - local higher res map
    mask_offset  - start index of mask in interest_map
    res_by_patch - size of global interest_map tiles compared to local mask - RESOLUTION_PATCH//dimp
    
    returns: interest_map
    '''
    interest_per_block = block_reduce(mask, block_size=res_by_patch)#meter by meter num pixels of interest
    
    local_indexes      = np.argwhere(interest_per_block) # indexes of local tiles where interest in non empty
    g_rows, g_cols     = (local_indexes + mask_offset).astype(int).T #global rows and columns with some interest

    new_interest_map   = np.copy(interest_map)
    new_interest_map[g_rows, g_cols]   = np.max([interest_map[g_rows, g_cols], interest_per_block[local_indexes.T[0], local_indexes.T[1]]], axis=0)
    
    return new_interest_map

# update_interest_map(interest_map, masks[0], (patch_mins-fl[0].min(axis=0)[:2])[i], RESOLUTION_PATCH//dimp)

def get_o3d_pcd_from_coordinates(np_locations, colors=None):
    '''`np_locations` (and optional `colors` 0-1) are np array with shape (n, 3). Can be obtained by int rgb / 255.'''
    pcd_locations        = o3d.geometry.PointCloud()
    pcd_locations.points = o3d.utility.Vector3dVector(np.array(np_locations))
    if not(colors is None):
        if len(colors) == 3:
            colors = np.repeat(colors, np_locations.shape[0]).reshape((3,-1)).T 

        pcd_locations.colors = o3d.utility.Vector3dVector(colors)
    return pcd_locations


#From segment anything exploratory code: show mask on an image with random color and transparet
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

####################################################################################################
##############################LANG SAM Display methods: #############################################
####################################################################################################
# Source:
# https://github.com/luca-medeiros/lang-segment-anything/blob/main/example_notebook/getting_started_with_lang_sam.ipynb
# import warnings
# import numpy as np
# import matplotlib.pyplot as plt
import requests
# from PIL import Image
from io import BytesIO
# from lang_sam import LangSAM

# image = "https://static01.nyt.com/images/2020/09/08/well/physed-cycle-walk/physed-cycle-walk-videoSixteenByNineJumbo1600-v2.jpg"
# image = "./image.jpg"
# text_prompt = "person on a bicycle"

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")