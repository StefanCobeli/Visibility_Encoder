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


def load_model_from_info_dict_path(info_dict_path):
    '''
    Load NeRFS trained model based on info dictionary.
    '''
    
    # Initialize NeRFS model with weights of trainedNeRF model
    info_dict       = pd.read_json(info_dict_path).to_dict()[0]
    print("Found the following non empty classes:\n\t", info_dict["non_empty_classes_names"])
    # return info_dict
    norm_params     = (torch.tensor(info_dict["xyz_centroid"]), torch.tensor(info_dict["xyz_max-min"]), torch.tensor(info_dict["xyzh_centroid"]), torch.tensor(info_dict["xyzh_max-min"]))

    trained_model_path = info_dict_path.replace("training_info_", "encoder_").replace(".json", ".pt")
    trained_encoder            = network.nerfs.NeRFS(norm_params=norm_params, surface_type="square", pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"],  view_dir_dim=info_dict["enc_input_size"])
    trained_encoder.load_state_dict(torch.load(trained_model_path))
    
    return trained_encoder, info_dict

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