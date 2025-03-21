import numpy as np
import pandas as pd
import seaborn as sns

import time
import torch

import os


from utils.geometry_utils import *

from utils.scripts.architectures.train_location_encoder import *
from utils.scripts.architectures.torch_nerf_src import network

semantics_to_colors = {
    'brick': 'red'
    ,'concrete':"grey", 'marble':"gold", 'plaster':"purple"
    , "metal": "forestgreen"
    ,"building" : "lightcoral" # Building color palette / ##DCDCDC
    , "water" : "aqua" # water color palette / #c4e2fe
    , "road" : "lavenderblush" # road color palette / #FFFFFF
    , "sidewalk" : "lightgoldenrodyellow" # sidewalk color palette / #EBEADF
    , "surface" : "lawngreen" # surface color palette / #CDDDB6
    , "tree" : "chartreuse" # Tree color palette / #9DB97F
    , "sky" : "deepskyblue" # Sky color palette  / #C7CFF4
    , "miscellaneous" : "violet" # miscelaneous color palette
}



def get_hidden_layer_predictions(data_path, trained_model_path, info_dict_path=None, frac=1, query_format=False):
    '''
    data_path - path to data file (test data dataframe) - as json file should work
    trained_model_path - trained nerf
    info_dict_path - if no path provided tries to match to an info_dict from the same folder.
    
    returns model_predictions, model_latent_features
    
    frac takes determinstically the first fraction of the data.
    '''
    model_folder = "/".join(trained_model_path.split("/")[:-1])
    
    test_df = pd.read_json(data_path)
    test_df = test_df.head(int(frac*test_df.shape[0]))
    
    if info_dict_path is None: # try to find info_dict in the same folder as the model
        try:
            model_name   = trained_model_path.split("/")[-1]
            model_version  = model_name.removesuffix(".pt").split("_")[-1]
            info_dict_path = f"{model_folder}/training_info_{model_version}.json"
        except:
            print("There was no training info_dict found for provided model. Please provide separate info_dict_path.")
            return

    # Initialize NeRFS model with weights of trainedNeRF model
    info_dict       = pd.read_json(info_dict_path).to_dict()[0]
    print("Found the following non empty classes:\n\t", info_dict["non_empty_classes_names"])
    # return info_dict
    norm_params     = (torch.tensor(info_dict["xyz_centroid"]), torch.tensor(info_dict["xyz_max-min"]), torch.tensor(info_dict["xyzh_centroid"]), torch.tensor(info_dict["xyzh_max-min"]))

    nerf_latent_features = []
    nerf_predictions     = []

    trained_encoder            = network.nerfs.NeRFS(norm_params=norm_params, surface_type="square", pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"],  view_dir_dim=info_dict["enc_input_size"])
    trained_encoder.load_state_dict(torch.load(trained_model_path))
    
    if "label_name" in info_dict:
        test_df["f_xyz"] = test_df[info_dict["label_name"]]
        
    print("Succesful model intialization and data loading.\nPredicting hidden layers...")
    #Get hidden layer for each sample in the dataset
    for i in tqdm(range(test_df.shape[0])):
        xyz  = torch.tensor(test_df.values[:,:6][i].astype(float)[:3])
        xyzh = torch.tensor(test_df.values[:,:6][i].astype(float)[3:])

        _, latent_features, prediction = trained_encoder.predict_from_raw(xyz, xyzh, return_latent_features=True)


        nerf_predictions.append(prediction.detach().numpy()[0])
        nerf_latent_features.append(latent_features.detach().numpy())
    #     break

    nerf_predictions     = np.vstack(nerf_predictions)
    nerf_latent_features = np.vstack(nerf_latent_features)
    
    print("Succesfully inferred hidden layer values for provided test set.")
    
    print("\nThe last prediction was:", prediction)
    print("The non empty classes are:", info_dict["non_empty_classes_names"])
    
    #return nerf_predictions, nerf_latent_features, test_df, trained_encoder
    if query_format:
        if test_df["f_xyz"].apply(lambda x: type(x)==str).values[0]:
            test_df["f_xyz_raw"] = test_df["f_xyz"].apply(eval)
            test_df["f_xyz"]     = test_df["f_xyz_raw"].apply(lambda d: [s/sum(d) for s in d]) 
        #else:
        #Assumes test_df["f_xyz"] is already normalized and between -1, 1 / with process_locations_visibility_data_frame already applied
        test_df["f_xyz"] = test_df["f_xyz"].apply(lambda fs:
                                                  {k : v for k, v in 
                                                   zip(info_dict["classes_names"], [(1 + f)/2 for f in fs]) 
                                                   if k in info_dict["non_empty_classes_names"]})
        
        #torch.nn.MSELoss()(desired_percentages, actual_percentages).mean()
        test_df["residual"]    = 0
        test_df["steps"]       = 0
        test_df["start_locs"]  = test_df.apply(lambda r: [r["x"], r["y"], r["z"]], axis=1)
        test_df["start_views"] = test_df.apply(lambda r: [r["xh"], r["yh"], r["zh"]], axis=1)
        
#     [dict(zip(info_dict["non_empty_classes_names"], d["predictions"][-1])) for d in debug_dicts] 
    
#     return nerf_predictions, nerf_latent_features, test_df
    return nerf_predictions, nerf_latent_features, test_df, trained_encoder
    


def test_encoder_on_data(data_path, model_path, model_version, missing_labels=False, batch_size=32, normalized_predictions="log", debugging_predictions=False):
    '''
    Testing model on location.csv file
    if missing labels, losses will be 0
    
    return mean_loss, all_losses, test_predictions, test_df, info_dict
    '''

    #2. read configuration of trained model to get normalizations
    info_dict       = parse_training_info(model_path, model_version)
    norm_params     = (info_dict["xyz_centroid"], info_dict["xyz_max-min"], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])


    #1. read trained model
    #print(model_path, model_version)
    trained_encoder = network.nerf.NeRF(pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"]\
        ,  view_dir_dim=info_dict["enc_input_size"], feat_dim=256)
    trained_encoder.load_state_dict(torch.load(f"{model_path}/encoder_{model_version}.pt"))

    #3. Reading locations data
    test_loc_path   = data_path

    test_df, _, _   = process_locations_visibility_data_frame(test_loc_path, norm_params, selected_label_indexes=info_dict["sli"], missing_labels=missing_labels)
    # print(test_df.columns)
    if "image_name" not in test_df:
        test_df["image_name"] = "no_image_name"
    if "f_xyz" not in test_df:
        test_df['f_xyz']      = "no_f_xyz"

    #1.Data loader from points
    test_dl  = get_location_visibility_loaders(test_df, missing_labels=missing_labels, only_test=True, batch_size=batch_size)


    #2. Encoder details
    _, criterion, optimizer, scheduler = get_location_visibility_encoder(info_dict["pos_enc_dim"], info_dict["num_present_classes"], feat_dim=256)
    #3. epoch from dataloaders:
    mean_loss, all_losses, test_predictions   = run_one_epoch_location_encoder(trained_encoder, criterion, optimizer\
                                           , test_dl, training_epoch=False, return_predictions=True, gt_labels=not(missing_labels))

    #Debuging predictions purposes:
    if debugging_predictions:
        test_df["f_xyz_gt"]         = test_df["f_xyz"]   #Value used as ground truth - loss was computed using this 
        test_df["predictions_raw"]  = test_predictions.tolist()   #Unnormalized predictions
        return mean_loss, all_losses, test_predictions, test_df, info_dict

    print(f"MSE on new predicted points locations:\n\t{mean_loss.mean()}")

    if normalized_predictions:
        test_predictions = get_normalized_distributions(test_predictions, norm_type=normalized_predictions)
    
    test_df["predictions"] = test_predictions.tolist()

    #Clean up the "csv" to be back in the original format:
    if "f_xyz_raw" in test_df:
        test_df["f_xyz"]       = test_df["f_xyz_raw"]
        test_df = test_df.drop("f_xyz_raw", axis=1)

    #Build Predictions path and save to new csv file.
    predictions_path = "/".join(test_loc_path.split("/")[:-1] + ["/predictions/"]) + f"{test_loc_path.split('/')[-1][:-4]}_with_predictions.csv"
    test_df = test_df.drop(["xn", "yn", "zn",	"xhn", "yhn", "zhn"], axis=1)
    # test_df.to_csv(predictions_path, index=False)
    # print("\n$$$$$$$$$$$$$$$$$$$$$\n")

    return mean_loss, all_losses, test_predictions, test_df, info_dict

# Moved from 2_Buildings_to_Exterior_Use_Case to test_location_encoder.py on 03.10.2025
def get_facade_predictions_as_tiles(base_points, building_height, info_dict_path, n_width = 5, n_samples = 50, discretization_type = "linear", verbose=True):
    '''n_width - tiles per smallest side
    n_samples - sampels per tile
    discretization_type - exponential or linear
    returns: facade_dicts_list, tile_dicsts_list 
        facade_dicts_list - points facade
        tile_dicsts_list  - tile / mesh facade
    '''
    #0. Load model
    # info_dict_path = "./utils/assets/data/semantics/models/training_info_bs_1024_1000.json"
    #info_dict_path = "./utils/assets/data/semantics/models/training_info_100.json"
    trained_encoder, info_dict = load_model_from_info_dict_path(info_dict_path, verbose)
    present_classes = info_dict["non_empty_classes_names"]

    #2. Builiding tiles
    from utils.geometry_utils import generate_vertical_squares
    from utils.geometry_utils import sort_polygon_points, compute_perpendicular_orientation
    
    #Fix height for all base points, to be the tallest one:
    # base_points[:,1] =  base_points[:,1].max() Not used -- #For some reason mixes the order of the sorted points
    # Adjust base points to be further away from the center by 1 percent:
    base_points = adjust_points(base_points, 20)

    # n_width = 5; n_samples = 50#50
    # n_height = 116; #points = [[2244.487116618618, -0.10353878972804864, 1112.2862697008927], [2279.1700691989004, -1.347170397506389, 1116.533378041281], [2251.2898400509403, 3.3851716251081143, 1057.75508162751], [2285.9727926312225, 2.141540017329774, 1062.0021899678982]]
    n_height = building_height
    points   = base_points

    centers, samples, side_length, side_ids = generate_vertical_squares(points, n_width, n_height, n_samples, natural_height=True, verbose=verbose) #the points are sorted inside generate_vertical_squares
    sorted_base_points            = sort_polygon_points(points)
    #side_length = 
    #Draw the computed centers and tile samples in o3d:
    # draw_facade_centers_and_tiles_in_o3d(points, centers, samples)            

    #3. Initialize facade_dict to be passed by server.py
    # camera_coordinates : [[x1,x2,...], [y1,y2,...], ..., [zh1,zh2,...]]
    #predictions: [p1,p2,...]
    #building: [b1,b2,...]; tree: [] ...
    # facade_dict={"camera_coordinates": [[],[],[],[],[],[]], "predictions":[]} 
    # facade_dict.update({s:[] for s in info_dict["non_empty_classes_names"]})
    #3. Initialize facade_dict (facade_dicts_list) to be passed by server.py
    # camera_coordinates : [x,y,z,xh,yh,zh], predictions:[l1,l2,...]
    facade_dicts_list = [] #"camera_coordinates", predictions, facede_side
    tile_dicsts_list  = [] #Each tile has "center", "dimension", "points", "mean_intensities"

    #Loop logs
    from tqdm.notebook import tqdm
    import numpy as np
    np.random.seed(1)
    prediction_dicts = []
    if verbose:
        print(f"The side length of each tile is {side_length} x {side_length}")
        print(f"\nPredicting visibility from raw xyz coordinates for:")
        print(f"\t{len(centers)} tiles x {len(samples[0])} locations per tile = {len(centers)*len(samples[0]):,} total facade location estimations.")

    #4. Iterate through tiles / centers
    import torch
    if verbose:
        progress_bar = tqdm(enumerate(centers), total=len(centers))
    else:
        progress_bar = enumerate(centers)

    for i, c in progress_bar:

        #i. Get side orientation given base points:
        #centers are list with the order [side1_c1, side1_c2,...side2_c1,...,side4_cn] - see generate_vertical_squares in geometry_utils.py
        centers_per_side = len(centers) // 4
        tile_side_id     = side_ids[i] #i // centers_per_side #index div centers on a side
        # print(f"{i}/{len(centers)}: Choosing side")
        bp1 = sorted_base_points[tile_side_id]
        bp2 = sorted_base_points[tile_side_id + 1] if tile_side_id < 3 else sorted_base_points[0]
        # print(f"Side base points: \n\t{bp1} - {bp2}")
        xyzh_normal      = compute_perpendicular_orientation(bp1, bp2)

        prediction_dicts.append([])
        tile_predictions = []
        tile_points      = []

        #For each tile / center predict all locations
        # print(f"{i}/{len(centers)}: iterating through center")
        for j, xyz_sample in enumerate(samples[i]):

            xyz  = torch.tensor(xyz_sample) #samples[i][j])
            #xyzh = torch.tensor(np.random.randint(0, 180, (3, ))) # to be replaced using perpendicular direction
            xyzh = torch.tensor(xyzh_normal)

            #xyzh = compute_perpendicular_orientation()

            _,_, prediction = trained_encoder.predict_from_raw(xyz, xyzh) #predictions are in tanh (-1, 1)

            percentage_predictions = (prediction / 2 + 0.5).detach().numpy()[0]
            predicition_dictionary = dict(zip(present_classes, percentage_predictions.tolist()))
            prediction_dicts[-1].append(predicition_dictionary)
            tile_predictions.append(percentage_predictions)

            #populate facade_dict to be passed by server.py
            camera_coordinates = np.hstack([xyz.detach().numpy().astype(int), xyzh.detach().numpy().astype(int)]).tolist()
            tile_points.append(xyz.detach().numpy().round(5).tolist())
            facade_dict = {"camera_coordinates":camera_coordinates, "predictions":percentage_predictions.round(5).tolist(), "facede_side": [bp1.tolist(), bp2.tolist()]}
            facade_dicts_list.append(facade_dict)
            # for i in range(6):
            #     facade_dict["camera_coordinates"][i].append(float(camera_coordinates[i]))
            # facade_dict["predictions"].append(float(prediction.detach().numpy()[0][-2]))#sky label index is -2
            # for i, s in enumerate(present_classes):
            #     facade_dict[s].append(float(percentage_predictions[i]))
        
        tile_dict = {}
        tile_dict["center"]            = centers[i].tolist()
        tile_dict["dimension"]         = side_length
        # tile_dict["points"]            = [centers[i]] + [bp1, bp2] + np.vstack(tile_points).tolist()
        # print(f"Side base points v2: \n\t{bp1.tolist()} - {bp2.tolist()}")
        tile_dict["points"]            = [bp1.tolist(), bp2.tolist()] + np.vstack(tile_points).tolist()
        tile_dict["mean_intensities"]  = dict(zip(present_classes, np.vstack(tile_predictions).mean(axis=0).tolist()))
        #maximums_per_class = dict(zip(tile_dict["mean_intensities"].keys(), np.vstack([list(tile_dict["mean_intensities"].values()) for t in tile_dicsts_list]).max(axis=0)))
        #tile_dict["colors"] = {s: intesity_to_color(i/maximums_per_class[s], discretization_type=discretization_type) \
        #                   for (s, i) in tile_dict["mean_intensities"].items()}
        tile_dicsts_list.append(tile_dict)

    if verbose:
        print("\n Iterated through all the centers.")
    
    # Add color to each tile based on normalized intesity
    maximums_per_class = dict(zip(tile_dicsts_list[0]["mean_intensities"].keys(), np.vstack([list(t["mean_intensities"].values()) for t in tile_dicsts_list]).max(axis=0)))
    for t in tile_dicsts_list:
        t["colors"] = {s: intesity_to_color(i/maximums_per_class[s], discretization_type=discretization_type) \
                        for (s, i) in t["mean_intensities"].items()}

    ##Normalize predictions to maximum of the predicted class:
    predictions_per_class = np.vstack([facade_dicts_list[i]["predictions"] for i in range(len(facade_dicts_list))])
    max_per_class = predictions_per_class.max(axis=0)
    # print("\nMaximums per class are:", max_per_class)
    # print("facade dict example:", facade_dicts_list[0])
    # print([p/max_per_class[i] for i,p in enumerate(facade_dicts_list[0]["predictions"])])
    for fd in facade_dicts_list: #Round predictions
        fd["predictions"] = [np.round(p / max_per_class[i], 5) for i,p in enumerate(fd["predictions"])]

    return facade_dicts_list, tile_dicsts_list# facade_dict


# info_dict_path = "./utils/assets/data/semantics/models/training_info_100.json"
# n_height = 20; points = [[2244.487116618618, -0.10353878972804864, 1112.2862697008927], [2279.1700691989004, -1.347170397506389, 1116.533378041281], [2251.2898400509403, 3.3851716251081143, 1057.75508162751], [2285.9727926312225, 2.141540017329774, 1062.0021899678982]]
   
# Moved from 2_Buildings_to_Exterior_Use_Case to test_location_encoder.py on 03.19.2025
def tuple_to_hexa_color(tuple_color):
    """https://stackoverflow.com/a/3380739
    tuple_color - ints in 0 - 255
    """
    if not(type(tuple_color[0]) is int):
        tuple_color = tuple([int(255 * c) for c in tuple_color])
        #print(tuple_color)
    return '#%02x%02x%02x' % tuple_color
    
# tuple_to_hexa_color(sns.color_palette("magma", n_colors=10)[0]), sns.color_palette("magma", n_colors=10)[0],\
# tuple_to_hexa_color(sns.color_palette("magma", n_colors=10)[-1]), sns.color_palette("magma", n_colors=10)[-1]

# Moved from 2_Buildings_to_Exterior_Use_Case to test_location_encoder.py on 03.19.2025 
def intesity_to_color(intensity, color_pallete=sns.color_palette("magma", n_colors=10), discretization_type="linear"):
    """
    intensity - subunitary between 0 and 1.
    discretization_type : linear or exponential, more prone to perceived visibility information
    """
    num_colors  = len(color_pallete)
    
    if discretization_type == "linear":
        color_value = min(int(intensity*num_colors), num_colors-1)
    if discretization_type == "exponential":
        exponential_factor = 1 - np.exp(-5 * intensity)
        color_value = min(int(exponential_factor*intensity*num_colors), num_colors-1)
        pass
    
    tuple_color = color_pallete[color_value]
    hex_color   = tuple_to_hexa_color(tuple_color)
    
    return hex_color

# len(sns.color_palette("magma", n_colors=10))
# intesity_to_color(.8)

def adjust_points(points, x):
    """
    Adjusts four 3D points by moving them x% away from their diagonally opposite point.
    
    Parameters:
    points (list of lists or np.array): A list or array of four 3D points.
    x (float): The percentage (0-100) of the distance to move away from the center.
    
    Returns:
    np.array: The adjusted points.
    """
    points = np.array(points)
    assert points.shape == (4, 3), "Input must be four 3D points."
    
    # Compute center of the four points
    center = np.mean(points, axis=0)
    
    # Define diagonally opposite pairs
    pairs = [(0, 3), (1, 2), (2, 1), (3, 0)]
    
    # Compute adjusted positions
    adjusted_points = np.copy(points)
    for i, (idx, opp_idx) in enumerate(pairs):
        direction = points[idx] - points[opp_idx]  # Direction away from opposite point
        distance = np.linalg.norm(points[idx] - center)  # Distance to center
        displacement = (x / 100) * distance * direction / np.linalg.norm(direction)  # Scale displacement
        adjusted_points[idx] += displacement  # Move the point
    
    return adjusted_points.tolist()
# Example usage:
# points = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# x = 10  # Move 10% away
# adjusted_points = adjust_points(points, x)
# print(adjusted_points)




def predict_facade_from_base_points(base_points, building_height, points_per_facade_face=100, normalized_predictions="log", batch_size=2**15, debugging_predictions=False):
    '''
    Given 4 input points, the height and the poitns per side geenerate locations, angles and predictions
    '''
    bp  = base_points
    ppf = points_per_facade_face
    bh  = building_height
    apl = 6 #angles per location

    #1. generate locations on the facade:
    full_facade, lbp = get_full_facade_from_basepoints(bp, building_height = bh, points_per_facade = ppf, seed=1)

    #a. Generate 6 Angles for each of the locations 
    gmm_path          = './utils/assets/models/angles_gmm_100.joblib'
    num_new_locations = full_facade.shape[0] * apl #6 angles per each location


    #b. Each location get repeated 6 times and gets a random camera angles sample
    facade_locations = full_facade.repeat(apl, axis=0)
    facade_angles    = generate_angles_with_gmm(num_new_locations, gmm_path)


    facade_np = np.hstack([facade_locations, facade_angles])

    #c. Save to new generated locations and agles to pandas csv

    facade_df = pd.DataFrame(facade_np, columns=["x", "y", "z", "xh", "yh", "zh"])
    facade_df["image_name"] = "no_image_name"
    facade_df['f_xyz']      = "no_f_xyz"

    new_building_name = "_".join(lbp.flatten().astype(str))
    new_building_path = f"./utils/assets/new_buildings/mockedFileName.csv"

    print(f"saved {len(facade_df)} new facade locations at:\n\t{new_building_path}")
    # facade_df.to_csv(new_building_path, index_label=False)
    facade_df.to_csv(new_building_path, index=False)

    #2. make predictions for each point on the facade:
    dp = new_building_path       # data path
    bs = batch_size# 2**14                   # batch size
    mp = "./utils/assets/models/"# path to models folder
    mv = 350 #model version

    mean_loss, all_losses, test_predictions, test_df, info_dict = \
    test_encoder_on_data(dp, mp, mv, missing_labels=True, batch_size=bs, normalized_predictions=normalized_predictions, debugging_predictions=debugging_predictions)
    #print(f"MSE on new predicted points locations:\n\t{mean_loss.mean()}")
    
    #Keep only the csv file with the predictions (saved in the test_encoder_on_data)
    # - Remove the csv transformed from the json request:
    os.remove(new_building_path)
    print(f"removed new facade csv locations to avoid memory surplus:\n\t{new_building_path}")
    facade_df["predictions"] = test_predictions.tolist() # test_df and facade_df might be the save - TODO: consider returning test_df and remove this line
    print()
    # facade_df.to_csv(new_building_path, index_label=False) #Saves index
    # facade_df.to_csv(new_building_path, index=False)       # does not save index - moved to test_encoder_on_data
    if debugging_predictions:
        return test_df
    
    return facade_df.drop(["image_name", "f_xyz"], axis=1)

def np_print_back_to_array(printed_np):
    '''
    parser for a printed numpy array
    return np_array
    '''
    lines       = printed_np.split('\n')
    #clean_lines = [l.strip(" []").replace("  ", " ").replace("  ", " ").split(" ") for l in lines]
    #https://stackoverflow.com/a/2077944/7136493
    clean_lines = [l.strip(" []").split() for l in lines]
    
    np_array    = np.vstack(clean_lines).astype(np.float32)
    
    return np_array
    
def parse_training_info(model_path=None, model_version=None, info_dict_path=None):
    '''
    Parse training details into dictionary.
    '''
    try:
        info_dict  = pd.read_json(info_dict_path).to_dict()[0]
        print(f"Succesfully read json config from {info_dict_path}")
        #print(info_dict)
        return info_dict
    except:
        try:
            # print(f"Reading json config {model_version}")
            info_dict  = pd.read_json(f"{model_path}/training_info_{model_version}.json").to_dict()[0]
            # print("Succesfully read json config from")
            # print(f"\t{model_path}/training_info_{model_version}.json")
            # print(info_dict)
            return info_dict
        except:
            # print("Reading csv config ")
            info_dict                     = pd.read_csv(f"{model_path}/training_info_{model_version}.csv", index_col=0).to_dict()["0"]
    
    #1. Parse available classes:
    info_dict["non_empty_classes_names"] = eval(info_dict["non_empty_classes_names"])[0][1:].split(" ")
    info_dict["classes_names"]           = eval(info_dict["classes_names"].replace(" ", ""))
    
    info_dict["non_empty_classes"]  = np.in1d(info_dict["classes_names"], info_dict["non_empty_classes_names"])
    info_dict["sli"]                = np.arange(len(info_dict["classes_names"]))[info_dict["non_empty_classes"]]

    info_dict["pos_enc_dim"] = eval(info_dict["pos_enc_dim"])
    info_dict["num_present_classes"] = eval(info_dict["num_present_classes"])
    info_dict["enc_input_size"]      = eval(info_dict["enc_input_size"])

    info_dict["final_training_loss"] = eval(info_dict["final_training_loss"])
    info_dict["final_test_loss"]     = eval(info_dict["final_test_loss"])

    info_dict["training_losses_summary"] = np_print_back_to_array(info_dict["training_losses_summary"])
    info_dict["training_losses_history"] = np_print_back_to_array(info_dict["training_losses_history"])
    info_dict["test_losses_summary"]  = np_print_back_to_array(info_dict["test_losses_summary"])
    info_dict["test_losses_history"]  = np_print_back_to_array(info_dict["test_losses_history"])
    
    #2. Parse normalization parameters:
    info_dict["xyz_centroid"]    = np.array(eval(info_dict["xyz_centroid"].replace("  ", ",")) )
    info_dict["xyz_max-min"]     = eval(info_dict["xyz_max-min"]) 
    
    #len > 2 - if number, floating point and digit after floating point
    info_dict["xyzh_centroid"]  = np.array([float(x) for x in info_dict["xyzh_centroid"].split(" ") if len(x)>2])
    info_dict["xyzh_max-min"]    = eval(info_dict["xyzh_max-min"]) 
    
    
    return info_dict


def get_normalized_distributions(predictions, norm_type="percentages", error_scaling=False):
    '''Normalize predictions between 0 and 1 to be used as input for seaborn color pallete'''
    #Log normalization:
    norm_preds      = predictions# preds[:, selected_label] # linear scaling, no logarithm
    # norm_preds      = np.log(1+1e-7+preds) #all labels log
    if norm_type == "log":
        norm_preds      = np.log(1+1e-7+predictions)
    
    #Linear normalization: 
    lin_norm_preds  = (norm_preds - norm_preds.min(axis=0)) / (norm_preds.max(axis=0) - norm_preds.min(axis=0))
    #lin_norm_preds  = (norm_preds - norm_preds.min(axis=0)) / 2 # Since tanh is the final activation max - min should be 1-(-1) = 2

    if norm_type == "percentages":
        lin_norm_preds = (predictions + 1) / 2 #Minimum of tanh is -1 maximum is 1
        #if there is a line with predictions that sum up to more than one. Cap them to maximum one, otherwise leave them as they were:
        overflow_ids                 = lin_norm_preds.sum(axis=1)>1
        lin_norm_preds[overflow_ids] = (lin_norm_preds / lin_norm_preds.sum(axis=1).reshape((-1,1)))[overflow_ids]    

    #Assums norm_preds are the actual MSE errors, not the predictions.
    if error_scaling:
        lin_norm_preds = norm_preds / 4 #Since Predictions are in (-1,1), maxium MSE is 4

    return lin_norm_preds

def get_pcd_with_color_intensity(xyz, normalized_intensities, color_pallete=sns.color_palette("coolwarm", as_cmap=True)):
    '''normalized between 0 and 1 normalized_intensities for each xyz'''

    intensity_colors = color_pallete(normalized_intensities * 255)[:, :3]
    #print(xyz.shape, intensity_colors.shape)
    
    pcd_intensity    = get_o3d_pcd_from_coordinates(xyz, intensity_colors)
    
    return pcd_intensity


'''Moved from Urban Perception Metrics Analysis.ipynb to test_location_encoder.py - March 17th 2025'''
def formula_and_dict_to_perception(formula_expression, f_xyz_semantics_dict):
    """
        Parse perception string formulas to value.
        Replace each semantic string in the formula_expression with it's value in f_xyz_semantics_dict
        Return the value of the evaluated formula.
    """
    form_literals = formula_expression.split(" ")
    for i in range(len(form_literals)):
        literal = form_literals[i] #on of the sematics
        if literal in f_xyz_semantics_dict:
            form_literals[i] = str(f_xyz_semantics_dict[literal])
    
    numerical_formula = " ".join(form_literals)
    try:
        index_value = eval(numerical_formula)
    except:#Division by 0
        index_value = 0
    return index_value #eval(numerical_formula)
    #print("Semantics: ", f_xyz_semantics_dict)
    #print("Semantics formula: ", formula_expression)
    #print("Numerical formula: ", numerical_formula)
    #print("Evaluated formula: ", eval(numerical_formula))



'''Moved from 1_Training_and_View_query_use_case to test_location_encoder.py to avoid circular imports not ./utils/scripts/architectures/train_location_encoder.py on Mar 17th 2025'''
def train_2d_projectors_and_clusteres(test_data_path, trained_model_path, projectors, clusterers, frac=1):
    '''
    Computing latent space, hidden feature clustering and 2d projections
        To dos in the full training function:
            train encoder model
            train and save as PKL: PCA, UMAP, GM, AGG, DBSCAN - num clusters (n and maybe 2n) - create folder for projectors.
            generate json with all these fields.
    
    '''
    #1. Get latent space
    nerf_predictions, nerf_latent_features, test_df, trained_encoder = \
        get_hidden_layer_predictions(test_data_path, trained_model_path, info_dict_path=None, frac=frac, query_format=True)#[2]
    
    # 2. Perform all projections:
    for projector in projectors:
        start_time = time()
        projector, projected_results_global = project_data_on_2d(global_data = nerf_latent_features, projector=projector, query_data=None)
        projector_name                      = projector.__class__.__name__
        projector_path                      = trained_model_path.replace("/encoder_", f"/{projector_name}_").replace(".pt", ".pkl")
        joblib.dump(projector, projector_path) 
        test_df[projector_name]             = projected_results_global.tolist()
        print(f"{projector_name} projection took {time() -  start_time:.2f}s")
        #print("Example projection: ", projected_results_global[0],"\n")
    
    # 3. Perform all clustering methods & zip assignment
    X               = nerf_latent_features
    for clusterer_name in clusterers:
        clusterer               = clusterers[clusterer_name]
        labels                  = clusterer.fit_predict(X)
        test_df[clusterer_name] = labels
        clusterer_path          = trained_model_path.replace("/encoder_", f"/{clusterer_name}_").replace(".pt", ".pkl")
        joblib.dump(clusterer, clusterer_path) 
        print(f"{clusterer_name} clustering took {time() -  start_time:.2f}s")
    
    test_df["zip"]      = locations_to_zip(test_df[["x", "y", "z"]].values.tolist()) #Same application method as in `gradient_walk_utils.py`
    location_types      = ["zip"] + list(clusterers.keys())
    test_df["clusters"] = test_df.apply(lambda r: {lid : r[lid] for lid in location_types}, axis=1)
    
    return test_df.drop(location_types, axis=1)




# from utils.scripts.architectures.train_location_encoder import train_model_on_data
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import umap
# import numpy as np
# import pandas as pd
from time import time
import joblib
from utils.gradient_walk_utils import project_data_on_2d, locations_to_zip
# from utils.test_location_encoder import get_hidden_layer_predictions#, load_model_from_info_dict_path
'''Moved from 1_Training_and_View_query_use_case to test_location_encoder.py to avoid circular imports not ./utils/scripts/architectures/train_location_encoder.py on Mar 17th 2025'''
def train_model_and_genrate_latent_space_projections(data_path, ne, sli, frac):
    '''
    input: 
        train (& test) data paths
        num_epochs
        sli : selected label indexes
        
    returns: jsons to be used in the interface
    '''

    #1. Train NeRF for ne epochs
    train_data_path = f"{data_path}/train.json"
    model_name      = f"encoder_{ne}.pt"
    encoder_net, tr_losses_history, test_losses_history, vdf \
    = train_model_on_data(data_path=train_data_path, num_epochs=ne, tsp=frac, selected_label_indexes=sli, model_name=model_name)

    # 2. train and save as PKL: PCA, UMAP, GM, AGG, DBSCAN - num clusters (n and maybe 2n) - create folder for projectors.
    test_data_path      = f"{data_path}/test.json"
    trained_model_path  = train_data_path.replace("train.json", f"models/{model_name}")
    projectors          = [PCA(n_components=2), umap.UMAP(n_components=2, n_neighbors=15)]
    clusterers         = {f"GM-{len(sli)}": GaussianMixture(n_components=len(sli), covariance_type='full', random_state=42),
                        f"AGG-{2*len(sli)}": AgglomerativeClustering(n_clusters=2*len(sli), linkage='ward'),
                        "DBSCAN": DBSCAN(eps=0.5, min_samples=max(5, vdf.shape[0]//100))
                       }
    test_as_query_df = train_2d_projectors_and_clusteres(test_data_path, trained_model_path, projectors, clusterers, frac=frac)
    
    #3. Generate jsons with query fields. If semantics case generate also perceptions json.
    test_json_path = trained_model_path[:trained_model_path.index("models/")] + "test_set_as_query.json"
    test_as_query_df.to_json(test_json_path, indent=4, orient="records")
    np.random.seed(1)
    test_as_query_df.sample(15).to_json(test_json_path.replace("test_set_as_query", "small_query"), indent=4, orient="records")
    print()
    print(", ". join([p.__class__.__name__ for p in projectors] + ["zip"] + list(clusterers.keys())\
                    ), f" test set as query, projectors and clusterers saved at: \n\t{test_json_path}")

    #3.b In semantics case save also a perception json
    if "tree" in test_as_query_df.f_xyz.values[0]: #Check if it is the semantics prediction case and generate perceptions:
        test_as_query_df_semantics = test_as_query_df.copy()
        perc_def = pd.read_json('./utils/assets/data/perception_metrics/predefinedPerceptions.json')
        predefined_formulas_dict = {p: perc_def[p].values[0]["expression"] for p in perc_def}
        from utils.test_location_encoder import formula_and_dict_to_perception
        
        test_as_query_df["f_xyz"] = test_as_query_df["f_xyz"].apply(lambda f_xyz:\
                {p: formula_and_dict_to_perception(predefined_formulas_dict[p], f_xyz) for p in predefined_formulas_dict})
         
        test_json_path = trained_model_path[:trained_model_path.index("models/")] + "test_set_perception_as_query.json"
        test_as_query_df.to_json(test_json_path, indent=4, orient="records")
        np.random.seed(1)
        test_as_query_df.sample(15).to_json(test_json_path.replace("test_set_perception_as_query", "small_query_perception"), indent=4, orient="records")

        print(", ". join([p.__class__.__name__ for p in projectors] + ["zip"] + list(clusterers.keys())\
                        ), f" perceptions test set as query, projectors and clusterers saved at: \n\t{test_json_path}")

        return (test_as_query_df_semantics, test_as_query_df), test_losses_history#semantics, perceptions dfs
    
    return (test_as_query_df, None), test_losses_history#buildings df, None - replicate perceptions return above


#moved to test_location_encoder.py from 1_encoder_experiment_training_density_requirements and gradient_walk_utils due to ciruclar import 03.18.2025 & in November 2024
def load_model_from_info_dict_path(info_dict_path, verbose=True):
    '''
    Load trained model based on path to info dictionary.
    '''
    # Initialize NeRFS model with weights of trainedNeRF model
    info_dict       = pd.read_json(info_dict_path).to_dict()[0]
    if verbose:
        print(f"Loading model as described in:\n\t{info_dict_path}")
        print("Found the following non empty classes:\n\t", info_dict["non_empty_classes_names"])
    # return info_dict
    norm_params     = (torch.tensor(info_dict["xyz_centroid"]), torch.tensor(info_dict["xyz_max-min"]), torch.tensor(info_dict["xyzh_centroid"]), torch.tensor(info_dict["xyzh_max-min"]))

    trained_model_path = info_dict_path.replace(".json", ".pt").replace("training_info", "encoder")
    trained_encoder            = network.nerfs.NeRFS(norm_params=norm_params, surface_type="square", pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"],  view_dir_dim=info_dict["enc_input_size"], verbose=verbose)
    trained_encoder.load_state_dict(torch.load(trained_model_path))
    
    return trained_encoder, info_dict
