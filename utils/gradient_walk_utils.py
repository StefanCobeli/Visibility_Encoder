
from cgi import test

import numpy as np
import pandas as pd
import seaborn as sns

import time
import torch

import os

from copy import copy

from utils.geometry_utils import *
from utils.geometry_utils import surface_parametric
from utils.test_location_encoder import parse_training_info

from utils.scripts.architectures.train_location_encoder import *
from utils.scripts.architectures.torch_nerf_src import network
# from utils.gradient_walk_utils import initialize_trained_encoder, initialize_input_as_tensor
# from utils.scripts.architectures.train_location_encoder import rescale_from_norm_params 

import joblib

#copied from ./development_notebooks/Building_Data_Analysis_bk_11212024.ipynb on Mon. 11.25.2024
def locations_to_zip(xyz_locations):
    '''
    Applies zip code to given xyz locations:
    xyz_locations: (n,3) numpy array of xyz locations
    
    returns zip_codes - np list of ints with zip codes for each location
    '''
    
    path_to_knn = "./utils/assets/data/zipcode_knn.pkl" # Change path when using it in server.py
    knn_loaded_test = joblib.load(path_to_knn)

    zip_codes = knn_loaded_test.predict(xyz_locations).astype(int)

    return zip_codes


def choose_model_based_on_query(desired_distribution):
    """
    tanh input and tanh output 
    [-1,1] -> [-1, 1]
    Find what model should be used.
    Return info_dict with model_path entry and 
    rectified query as list of length being the output of the model.
    
    rectified_query: -1 where nothing specified and value where specified.

    return info_dict, rectified_query
    """
    query_labels = None
    print(f"The received query is:\n\t{desired_distribution}")
    #Scenarios:
    # 0. Custom Perception
    custom_formula_strings = []
    custom_formula_names   = []
    if type(list(desired_distribution.values())[0]) is dict:
        semantics_full = ['building', 'water', 'road', 'sidewalk', 'surface', 'tree', 'sky', "miscellaneous"]
        #Label names - e.g. enclosure, walkability, etc.
        custom_formula_names   = list(desired_distribution.keys())
        #formula strings - e.g. sidewalk / (road + sidewalk)
        custom_formula_strings = [list(desired_distribution[cfn].keys())[0] for cfn in custom_formula_names]
        rectified_distribution = [desired_distribution[cfn][cfs] for cfn, cfs in zip(custom_formula_names, custom_formula_strings)]

        #same model and info_dict as in the #semantics_full case
        # info_dict_path = "./utils/assets/data/full_semantics/models/training_info_1000.json" #old full path with only two height levels
        # info_dict_path = "./utils/assets/data/semantics_final/models/training_info_1000.json"#final data with six height levels and all 8 semantic classes # December 2024
        info_dict_path = "./utils/assets/data/semantics/models/training_info_100.json"

        print("\nCustom perception query case found:")
        print(f"Querying for custom perceptions: \n\t{custom_formula_names}\nwith forumulas:\n\t{custom_formula_names}")
        query_labels = custom_formula_names

    else:
        #Original style query either {"water":d1, "sky":d2, ...} or [d1, d2, ...]
        #Semantics
        #1.  ['building', ' water', 'tree', 'sky']
        semantics_pricipal   = ['building', 'water', 'tree', 'sky'] 
        labels, label_ids, query_ids = np.intersect1d(semantics_pricipal, list(desired_distribution.keys()), return_indices=True)
        if len(labels) > 0: #semantics_pricipal case
            info_dict_path = "./utils/assets/data/splits_physical/models/training_info_100.json"
            rectified_distribution = np.zeros_like(semantics_pricipal, dtype=float) #- 1
            rectified_distribution[label_ids] = [desired_distribution[semantics_pricipal[qi]] for qi in label_ids]

            print(labels, label_ids, query_ids)
            print(f"Querying for principal semantics:")
            query_labels = semantics_pricipal

        #2.  [' building' ' water' ' road ' ' sidewalk' ' surface' ' tree' ' sky']
        semantics_adders     = ['road', 'sidewalk', 'surface']
        # semantics_full       = ['building', 'water', 'road', 'sidewalk', 'surface', 'tree', 'sky'] #Old full semantics before unifying perception and semantics 11.25.2024
        semantics_full = ['building', 'water', 'road', 'sidewalk', 'surface', 'tree', 'sky', "miscellaneous"]
        labels, label_ids, query_ids = np.intersect1d(semantics_adders, list(desired_distribution.keys()), return_indices=True)
        if len(labels) > 0: #semantics_full case
            labels, label_ids, query_ids = np.intersect1d(semantics_full, list(desired_distribution.keys()), return_indices=True)
            # info_dict_path = "./utils/assets/data/full_semantics/models/training_info_1000.json" #old full path with only two height levels
            # info_dict_path = "./utils/assets/data/semantics_final/models/training_info_1000.json"#final data with six height levels and all 8 semantic classes # December 2024
            info_dict_path = "./utils/assets/data/semantics/models/training_info_100.json"
            rectified_distribution = np.zeros_like(semantics_full, dtype=float)  #- 1
            rectified_distribution[label_ids] = [desired_distribution[semantics_full[qi]] for qi in label_ids]
            print("\nFull semantics found:")
            print(f"Querying for full semantics: \n\t{semantics_full}")
            query_labels = semantics_full

        #Perception:
        #3. ["greenness", "openness", "imageability", "encolusre", "walkability", "serenity"]
        perceptions          = ["greenness", "openness", "imageability", "enclosure", "walkability", "serenity"]
        labels, label_ids, query_ids = np.intersect1d(perceptions, list(desired_distribution.keys()), return_indices=True)
        if len(labels) > 0: #perception case
            info_dict_path = "./utils/assets/data/perception_metrics/models/training_info_100.json"
            rectified_distribution = np.zeros_like(perceptions, dtype=float) #- 1
            rectified_distribution[label_ids] = [desired_distribution[perceptions[qi]] for qi in label_ids]
            print(f"Querying for perception semantics: \n\t{perceptions}")
            query_labels = perceptions
        

    #Building Materials:
    #4. ---
    #Combinations of any values - semantics, perception and buildings:


    # model_folder = "/".join(trained_model_path.split("/")[:-1])
    # model_name   = trained_model_path.split("/")[-1].replace("training_info", "encoder").replace("json", "pt")
     # trained_encoder.load_state_dict(info_dict["model_path"])
    print(f"The query was:\n\t{desired_distribution}")
    print(f"Looking for the rectified query:")
    print(f"\t{dict(zip(query_labels, rectified_distribution))}")
    print("~The rectified query will be further tanh normalized (-1, 1) in the nerfs.EncoderNeRFSDataset constructor.\n")

    #print(info_dict_path)
    info_dict = parse_training_info(info_dict_path=info_dict_path)
    info_dict["model_path"] = info_dict_path.replace("training_info", "encoder").replace("json", "pt")
    
    #pass also the information regarding the pontential custom formulas:
    info_dict["custom_formula_names"]   = custom_formula_names
    info_dict["custom_formula_strings"] = custom_formula_strings


    return info_dict, rectified_distribution # to be passed to gradient_walk_on_surface -  used for norm params and model path.

def initialize_trained_encoder(encoder_name="semantics"):
    '''
        Intialize encoder from hard coded file path and return also the training information in info_dict
        trained_encoder, info_dict = initialize_trained_encoder()
        encoder_name - {"semantics", "perception", "buildings", "general"}
    '''
    
    mv = 100   #350 - trained with 4 sematic classes                  #model version
    # mv = 1000 #1000 - trained with 7 sematic classes and 10 depth estimators
    mp = "./utils/assets/data/semantics/models"# path to models folder
    info_dict       = parse_training_info(mp, mv)
    trained_encoder = network.nerf.NeRF(pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"]\
        ,  view_dir_dim=info_dict["enc_input_size"], feat_dim=256)
    trained_model_path = f"{mp}/encoder_{mv}.pt"
    print(f"Loading model from {trained_model_path}")
    trained_encoder.load_state_dict(torch.load(trained_model_path))
    
    return trained_encoder, info_dict

def query_locations_on_surface(desired_distribution, surface_basis, surface_type, num_locations=10, search_intervals=np.ones(4) * .2, lt=.01, lrate=10, max_steps = 100, seed=1):
    '''Similar in api to query_locations
    Returns the same dataframe al_df for easy server posting'''
    
    seed=1
    set_seed(seed)
    print("Random seed:", seed)
    # np.random.seed(seed)

    
    # Check if desired_distribution is list or dictionary:
    #TODO: Extract desired distrubutions from potential dictionary
    #1. Make a decision of encoder to be used based on the provided keys:
    # encoder_name, info_dict = choose_model_based_on_query()
    original_distribution = copy(desired_distribution)
    if type(desired_distribution) is dict: 

        #Make search intervals based on the length of desired_distribution
        info_dict, rectified_distribution = choose_model_based_on_query(desired_distribution)
        interval = search_intervals[0]
        search_intervals = np.where(rectified_distribution==-1, 10, np.ones_like(rectified_distribution) * interval)

        desired_distribution = torch.tensor(rectified_distribution).to(torch.float32)

        print(f"Search Intervals:\n\t{search_intervals}")
        print(f"Detected custom formula strings:\n\t{info_dict['custom_formula_strings']}")

    if type(desired_distribution) is list:
        desired_distribution = torch.tensor([float(x) for x in desired_distribution]).to(torch.float32)
        info_dict = None
    #print(desired_distribution)
    
    n_trials  = num_locations * 5#2 #10
    #Needed just for view_dir intialization
    print("\nProcessing locations_example data for untilted viewing directions. Locations are sampled randomly.")
    vis_df, normp, _   = process_locations_visibility_data_frame("./utils/assets/test_data/locations_example.csv")  
    locs_array         = vis_df.values[:,:6].astype(float)
    
    ach_locs    = [] #achieved locations
    debug_dicts = []

    #adjust learning rate to the dimensions of the surface. The learning rate should not be larger than 10e-5 agains plane dim.
    print("surface dimensions:", surface_basis[2])
    minimum_surface_dimension = surface_basis[2].abs().min()
    lrate_surface_ratio      = lrate / minimum_surface_dimension #2500 / (5*1e-3)
    print("Learning rate, surface dim ratio:", lrate_surface_ratio)
    lrate_ratio_threshold = 10e-2
    if lrate_surface_ratio > lrate_ratio_threshold: #if lrate is to big make it smaller.
        print(f"Default lrate: {lrate}")
        lrate = minimum_surface_dimension * lrate_ratio_threshold
        print(f"Adjusted lrate to {lrate}, {lrate / minimum_surface_dimension}")

    for i in tqdm(range(n_trials)):
        
        #To optimize viewing directions: requires_grad=True - Freeze view direction  requires_grad=False
        #Change also  self.view_dir.retain_grad() in EncoderNeRFSDataset
        #Get a random view direction / angle from the training set.
        # view_dir           = torch.tensor(locs_array[np.random.randint(locs_array.shape[0])][3:], requires_grad=True)
        view_dir           = torch.tensor(locs_array[np.random.randint(locs_array.shape[0])][3:], requires_grad=False)
        a, b               = torch.tensor(np.random.random()), torch.tensor(np.random.random()) #torch.rand(2) 
        parameters         = (a, b)
        # print(a, b)
        
        raw_pos, debugging_dict = gradient_walk_on_surface(parameters, view_dir, desired_distribution, surface_basis, surface_type\
                 , intervals=search_intervals, n_steps=max_steps, loss_threshold=lt, lrate=lrate, debugging_return=True, verbose=False\
                 , info_dict=info_dict)
        # debugging_dict["predictions"] - are the actual_percentages [0, 1] for the whole trajectory
        # desired_distribution - are percentages [0, 1]

        ##################################################################################
        # Check if prediction is in +- distance from desired distribution
        # desired_prediction      = desired_distribution.detach().numpy() #* 2 - 1
        # actual_prediction       = np.array(debugging_dict["predictions"])[-1] #* 2 - 1
        
        # admisable_interval      = 0.1 # admisable interval in [-1, 1] tanh deviation / twice as it would be in percentages [0, 1]
        
        # close_prediction      = np.logical_or(desired_prediction < 0.01, np.abs(desired_prediction - actual_prediction) < admisable_interval)

        # print("\ndesired_prediction:", np.round(desired_prediction, 2), "\nactual_prediction:", np.round(actual_prediction, 2)\
        # , "\nclose_prediction:", close_prediction, "\n") 

        # print("Not close prediction sum:", np.logical_not(close_prediction), np.logical_not(close_prediction).sum())
        # if np.logical_not(close_prediction).sum() < 2:
        #     print(f"Prediction  {i+1} is close enough:")
        ##################################################################################

        admissible_interval = 0.1
        admissible_errors   = 1
        if debugging_dict["close_enough"] != -1:
            print(f"Detected close enough location {i+1}.")
            #break
        else:
            print(f"Skipped locations {i+1}")
            continue

        
        ach_locs.append(np.hstack([raw_pos, debugging_dict["last_view_dir"]]))
        debug_dicts.append(debugging_dict)
        
        if len(ach_locs) >= num_locations:
            break 
        
    al_df               = pd.DataFrame(data=ach_locs, columns=vis_df.columns.values[:6])#achived locations data frame
    
    # print(list(info_dict.keys()))
    # print(info_dict["non_empty_classes_names"])

    #In case custom formula names were used, change the class names:
    if len(info_dict["custom_formula_names"]) > 0:
        info_dict["non_empty_classes_names"] = info_dict["custom_formula_names"]

    # print(info_dict["non_empty_classes_names"])

    # print("Features available for each location:", list(debug_dicts[-1].keys()))
    # print("Example of returned predictions - f_xyz:", debug_dicts[0]["predictions"][0])

    # Assemble JSON to be outputed by server:
    if info_dict is None:
        al_df["f_xyz"]      = [d["predictions"][-1] for d in debug_dicts] # Originally returned as list of values
    else:
        al_df["f_xyz"]      = [dict(zip(info_dict["non_empty_classes_names"], d["predictions"][-1])) for d in debug_dicts] # Return as dictionary of values.
    al_df["residual"]   = [d["final_residual"] for d in debug_dicts]
    al_df["steps"]      = [len(d["trajectory"]) for d in debug_dicts]
    al_df["start_locs"] = [d["trajectory"][0] for d in debug_dicts]
    al_df["start_views"] = [d["trajectory_view_dir"][0] for d in debug_dicts]

    print("Search surface had the parameters:\n\t", surface_basis)

    print(f"The original query was:\n\t{original_distribution}")
    print(f"The rectified query was:\n\t{desired_distribution}")

    ###print status of returned results
    al_df["close_enough"] = [d["close_enough"] for d in debug_dicts]
    close_enough_dictionary = dict(al_df.groupby("close_enough").count()["f_xyz"])
    for num_errors in close_enough_dictionary:
        print(f"{close_enough_dictionary[num_errors]} locations -> {num_errors} missed labels.")

        missed_trials       = (i + 1) - len(al_df) #n_trials - len(al_df)
        not_found_locations = num_locations - len(al_df) #n_trials - len(al_df)
        print(f"\nFound       {len(al_df)} / {num_locations} requested locations;")
        print(f"An additional {not_found_locations} were requested;")
        print(f"Missed        {missed_trials} / {i+1} trials.\n")
    print("Result misses status:",  dict(al_df.groupby("close_enough").count()["f_xyz"]), "\n")

    if len(al_df)==0:
        return al_df

    #Assign zip locations: see ./development_notebooks/Building_Data_Analysis_bk_11212024.ipynb for the application on the whole test set:
    al_df["zip"] = locations_to_zip(al_df[["x", "y", "z"]].values.tolist())

    #2D projections:
    if "final_latent_features" in debug_dicts[-1].keys():
        # al_df["final_latent_features"] = [d["final_latent_features"] for d in debug_dicts]
        nerf_latent_features = np.vstack([d["final_latent_features"] for d in debug_dicts])
        print("The shape of latent features is:", nerf_latent_features.shape)

        al_df["PCA"] = use_fitted_projector(nerf_latent_features, "PCA", info_dict["model_path"])
        al_df["UMAP"] = use_fitted_projector(nerf_latent_features, "UMAP", info_dict["model_path"])
    
    

    return al_df.sort_values("residual")


def gradient_walk_on_surface(parameters, view_dir, desired_target, surface_basis, surface_type, intervals=np.ones(4) * .1, n_steps=10, loss_threshold=0.1, lrate=5*1e-2, debugging_return=True, verbose=True, info_dict=None):
    ''' 
    Gradient walk for only one location.
    parameters - (init_a, init_b) between 0 and 1
    view_dir - xyzh between -180, 180.
    desired_target - normalized percentages [0, 1] - normalizetion to tanh -1,1 happens in EncoderNeRFSDataset
    
    returns either
        raw_pos, perc_pred
    or if debugging_return:
        raw_pos, deb_dict
    '''
    #Intial parameters
    init_a, init_b = parameters #TODO make intial a and b to be according to designated location / Implement inverse transformation. xyz -> ab
    # print(init_a, init_b)
    p, c, r        = surface_basis
    #0. Load trained model
    #TODO: pcr to be mixed in a single paramters variable.
    if info_dict is None:
        _, info_dict = initialize_trained_encoder()
        info_dict["model_path"] = f"./utils/assets/models/encoder_350.pt"

    #Added with custom formulas:
    custom_formula_strings = info_dict["custom_formula_strings"]

    norm_params                = (torch.tensor(info_dict["xyz_centroid"]), torch.tensor(info_dict["xyz_max-min"]), torch.tensor(info_dict["xyzh_centroid"]), torch.tensor(info_dict["xyzh_max-min"]))
    trained_encoder            = network.nerfs.NeRFS(p, c, r, norm_params, surface_type=surface_type, pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"],  view_dir_dim=info_dict["enc_input_size"], custom_formula_strings=custom_formula_strings)
    
    trained_encoder.load_state_dict(torch.load(info_dict["model_path"]))
    # trained_encoder.load_state_dict(info_dict["model_path"])
    # trained_encoder.load_state_dict(torch.load(f"./utils/assets/models/encoder_1000.pt"))
    criterion                  = torch.nn.MSELoss(reduction='none')
    
    #0. Load data to torch dataset
    #TODO: remove pcr from dataset condructor and surface_type
    # print("desired_target:", desired_target)
    sample_batch     = network.nerfs.EncoderNeRFSDataset(init_a, init_b, p, c, r, view_dir, desired_target, "square", norm_params)[0]
    
    input_a, input_b, input_dir = sample_batch["a"], sample_batch["b"], sample_batch["view_dir"]
    # print(init_a, init_b)
    # lrate      = 5*1e-2
    optimizer  = torch.optim.Adam(params=[input_a, input_b, input_dir], lr=lrate)
    
    gradients_norm  = []; predictions = []; trajectory = []; loss_trajectory = []; inputs = []; trajectory_view_dir = []#maybe trajectory view if needed
    if verbose:
        parsing_bar     = tqdm(range(n_steps))
    else:
        parsing_bar = range(n_steps)
    
    print()
    jumped_off_surface_step = -1 #If the predictions didn't get off surface then it's -1, otherwise the i where it got off surface
    close_enough            = -1 # number of missed labels -1 means above admissible / Flag to detect if location was close, to be returnd in debugging_dict
    for i in parsing_bar:
        # print(f"Optimization step {i}/{n_steps}")

        inputs.append((input_a.detach().numpy().tolist(), input_b.detach().numpy().tolist(), input_dir.detach().numpy().tolist()))
        ##### a. Predict output distribution
        (raw_pos, raw_view), latent_features, output = trained_encoder(input_a, input_b, input_dir, return_latent_features=True)
        #old version prediction witout latent features - changed on 11/11/2024
        # raw_pos, raw_view, output     = trained_encoder(input_a, input_b, input_dir)
        prediction = (output.detach().numpy()); labels = sample_batch["output"]
        
        perc_pred         = (prediction[0] + 1) / 2
        predictions.append(perc_pred)
        trajectory.append((raw_pos.detach().numpy()))
        trajectory_view_dir.append(raw_view.detach().numpy())


        # print("Intervals:", intervals)
        #Adaptive labels to interval:
        #TODO: Handle case where prediction has different dimensions compared to the labels. 
        # Front_end should not only the output values but also their lables, e.g. [0,1,0,0] - [b, w, t, s]. 
        if intervals is not None: 
            # print("\nOutput:", output.detach().numpy(), "\tSpecified labels:", labels)
            # print("\nprediction, labels.numpy(), intervals")
            # print(prediction, labels.numpy(), intervals)
            interval_target = interval_desired_target_loss(prediction, labels.numpy(), intervals)
            # print("labels, interval_target, labels - interval_target")
            # print(labels, interval_target, labels - interval_target)
            labels          = interval_target
            # print("Output:", output.detach().numpy(), "\tRectified labels:", labels)

        ##### b. Compute loss
        loss      = criterion(output, labels)

        ##### c. Gradient step using optimizer:
        optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        #loss.mean().backward()
        a_grad, b_grad, dir_grad = input_a.grad, input_b.grad, input_dir.grad
        optimizer.step()
        if not((0 < input_a.item() < 1) and (0 < input_b.item() < 1)):
            print(f"Jumped out of the surface on step {i}/{n_steps}, with a: {input_a.item()} and b: {input_b.item() }")
            jumped_off_surface_step = i
            break
        #print(raw_view.detach().numpy())
        ##### d. Log found gradients and predictions as percentages
        loss_trajectory.append(loss.detach().numpy())

        pos_grad_norm     = (np.linalg.norm(a_grad), np.linalg.norm(b_grad))#, np.linalg.norm(dir_grad))
        gradients_norm.append(pos_grad_norm)
        
        if verbose:
            parsing_bar.set_description(f"Gradient norm: a {pos_grad_norm[0]:.4f}, b {pos_grad_norm[1]:.4f}, dir {pos_grad_norm[2]:.4f}")


        admissible_interval = 0.09 #tanh interval. Error will be double.
        admissible_errors   = 1
        close_enough = prediction_is_close_enough(desired_prediction=labels.detach().numpy() / 2 + 0.5\
                                    , actual_prediction=perc_pred\
                                    , admissible_interval=admissible_interval\
                                    , admissible_errors=admissible_errors)
        if close_enough != -1: #This means it is close enough, by #close_enough errors.
            # print("Detected close enough prediction.")
            break

        if loss.mean() < loss_threshold:
            print(f"Loss {loss.mean()} passsed the threshold {loss_threshold}. Stoping the optimization.")
            break
    
           
    desired_percentages = (sample_batch["output"] + 1) / 2
    actual_percentages  = torch.Tensor(predictions[-1])
    # if MSE residual:
    final_residual = criterion(desired_percentages, actual_percentages).mean().detach().item()
    # # if RMSE residual:
    # final_residual = torch.sqrt(criterion(desired_percentages, actual_percentages)).mean().detach().item()
    #print(final_residual, desired_percentages, actual_percentages.detach().numpy().round(2))
    
    raw_pos = trajectory[-1]
    if debugging_return:
        debugging_dict = {"final_residual":final_residual, "trajectory":trajectory, "last_view_dir":raw_view.detach().numpy(), "predictions":predictions, "jumped_off_surface_step":jumped_off_surface_step, "close_enough":close_enough, "gradients_norm":gradients_norm, "loss_trajectory":loss_trajectory, "inputs":inputs, "trajectory_view_dir":trajectory_view_dir, "final_latent_features":latent_features.detach().numpy()}
        return raw_pos, debugging_dict
    else:
        return raw_pos, perc_pred

##### 2D latent space projections:
def project_data_on_2d(global_data, projector=None, query_data=None):
    '''
    global_data: (n, hidden_dim) - projector trainging features - NN hidden layer 
    prjoector: sklearn 2d projector - PCA, TSNE, UMAP
    new_data: (m, hidden_dim) - projector transform features - NN hidden layer 
    '''

    from sklearn.utils.validation import check_is_fitted
    #if projector.__class__.__name__ == "TSNE":##TODO / TSNE does not have .transform()
    
    try:
        check_is_fitted(projector)
        print(f"{projector.__class__.__name__} already fitted.")
        projected_results_global = projector.transform(global_data)
    except:
        print(f"Fitting not fitted {projector.__class__.__name__}...")
        projected_results_global = projector.fit_transform(global_data)
    
    if query_data is not None:
        projected_results_query = projector.transform(query_data)
        return projector, (projected_results_global, projected_results_query)
    

    return projector, projected_results_global

def use_fitted_projector(latent_features, projector_name, model_path):
    '''
    Project latent features into 2d using trained 2d projector taken from the same path as a trained model.
    latent_features: (n, n_features) numpy array
    projector_name: UMAP or PCA
    model_path: path to trained encoder model
    '''
    
    projector_path         = model_path.replace("/encoder_", f"/{projector_name}_").replace(".pt", ".pkl")
    projector              = joblib.load(projector_path)
    projector, projections = project_data_on_2d(latent_features, projector)

    print(f"{projector.__class__.__name__} projection -- OK.")
    
    return projections.tolist()



#######
###### Querying without Surface:
######

def query_locations(desired_distribution, num_locations=10, search_intervals=np.ones(4) * .2, lt=.01, at=10, max_steps = 100, seed=1, debugging=False):
    '''
    search_intervals:  
    lt:                loss threshold
    at:                acceptable factor loss; if loss is at least at * lt then consider the location
    return num_locations with desired_distribution plus minus the search_intervals with a loss threshold smaller than lt'''
    
    np.random.seed(seed)
    
    vis_df, normp, _   = process_locations_visibility_data_frame("./utils/assets/test_data/locations_example.csv")  
    locs_array         = vis_df.values[:,:6].astype(float)
    targets_array = (np.vstack((vis_df["f_xyz"]))).tolist()
    
    num_ps      = [] #number performed_steps
    mean_losses = [] # final mean losses for each pair input target
    residuals   = [] # MSE between desired and actual percentages
    targets     = [] #(achieved, desired)
    locations   = [] #(start, achieved)
    spatial_differences = [] #(initial, final)

    n_trials  = num_locations
    #max_steps = 100

    custom_target_d = desired_distribution #Distribution in percentages 
    desired_target  = (np.array(custom_target_d) * 2 - 1).tolist() #Distribution in tanh
    
    # original_distribution = copy(desired_distribution)
    # if type(desired_distribution) is dict: 

    #     #Make search intervals based on the length of desired_distribution
    #     info_dict, rectified_distribution = choose_model_based_on_query(desired_distribution)
    #     interval = search_intervals[0]
    #     search_intervals = np.where(rectified_distribution==-1, 10, np.ones_like(rectified_distribution) * interval)

    #     desired_distribution = torch.tensor(rectified_distribution).to(torch.float32)

    #     print(f"Search Intervals:\n\t{search_intervals}")
    #     print(f"Detected custom formula strings:\n\t{info_dict['custom_formula_strings']}")
    
    # desired_target = desired_distribution


    for i in tqdm(range(n_trials * 10)):
        
        crt_id = np.random.randint(0, len(locs_array))
        
        #des_id = np.random.randint(0, len(targets_array))
        #rd_ids += [crt_id, des_id]

        actual_loc    = locs_array[crt_id]
        actual_target = targets_array[crt_id]

        ### Gradient Walk: - For each location until finding num_locations.
        # print("Intervals:", search_intervals)
        achieved_loc, perc_pred, tr, gn, prds, lstr, ps, fr = gradient_walk(actual_loc, desired_target, search_intervals, max_steps, lt, True)

        print("Achieved location:", achieved_loc)
        if achieved_loc[1] < 10: #If location is below the scene.
            continue

        if lstr[-1].mean() < (at * lt): #if last loss is lower than - theshold x acceptaple factor # loss trajectory
            locations.append((actual_loc, achieved_loc))

            num_ps.append(ps)
            mean_losses.append(lstr[-1].mean())
            residuals.append(fr)

            start    =  [(at+1)/2 for at in actual_target]
            achieved =  perc_pred
            desired  =  [(dt+1)/2 for dt in desired_target]
            targets.append([np.round(t, 2) for t in [achieved, desired]])
        
        if len(locations) >= num_locations:
            break

    #print(vis_df.columns.values)    #print(vis_df.iloc[0])   #print(pd.DataFrame(data=locations[0], columns=vis_df.columns.values[:6]))
    
    ach_locs = [l[1] for l in locations]
    al_df    = pd.DataFrame(data=ach_locs, columns=vis_df.columns.values[:6])#achived locations data frame
    
    al_df["f_xyz"]     = [t[0] for t in targets]
    al_df["residual"]  = residuals
    al_df["steps"]     = num_ps
    al_df["start_locs"] = [l[0][:3] for l in locations] #only xyz locations without angles
    
    return al_df.sort_values("residual")


def gradient_walk(actual_loc, desired_target, intervals=None, n_steps=10, loss_threshold=0, debugging_return=False, optimizer=None, verbose=False):
    '''
    Go from actual_loc to location looking like desired_target
    actual_loc     - x,y,z,zh,yh,zh; - not normalized
    desired_target - [b,w,t,s] - tanh normalized -1,1 / according to the default model in - initialize_trained_encoder()
    if debugging_return:
        return actual_loc, perc_pred, trajectory, gradients_norm, predictions, loss_trajectory, performed_steps
    return actual_loc, perc_pred
    '''
    
    criterion                  = torch.nn.MSELoss(reduction='none')
    trained_encoder, info_dict = initialize_trained_encoder()

    sample_batch    = initialize_input_as_tensor(actual_loc, desired_target, info_dict)

    trajectory      = [np.array(actual_loc)]
    gradients_norm  = []
    predictions     = []
    loss_trajectory = []
    residuals       = []


    input_pos = torch.autograd.Variable(sample_batch["input_pos_raw"], requires_grad=True)
    input_dir = sample_batch["input_dir_raw"]

    #loss_threshold = np.inf #.45
    if optimizer is None:
        lrate      = 1e-2#.05#1e-2#.05 #1e-2
        # optimizer  = torch.optim.Adam(params=[input_pos, input_dir], lr=lrate)
        optimizer  = torch.optim.Adam(params=[input_pos], lr=lrate)#Optimize just the position

    #n_steps        = 100
    if verbose:
        parsing_bar    = tqdm(range(n_steps))
    else:
        parsing_bar    = range(n_steps)

    for i in parsing_bar:

        #TODO: project input_pos (and input_dir) on search space.

        ##### a. Predict output distribution
        output     = trained_encoder(input_pos, input_dir, from_raw=True)
        prediction = (output.detach().numpy())
        labels     = sample_batch["output"]
        # print(f"Predictions - {prediction.shape}: {prediction}, \nlabels - {labels.shape}: {labels.numpy()}, \nintervals: {intervals}")

        # print("Intervals:", intervals)
        #Adaptive labels to interval:
        if intervals is not None:
            interval_target = interval_desired_target_loss(prediction, labels.numpy(), intervals)
            #print(np.round(prediction, 2), np.round(labels, 2), np.round(interval_target, 2))
            labels          = interval_target
            # print(labels, interval_target)

        ##### b. Compute loss
        loss      = criterion(output,labels)

        ##### c. Gradient step using optimizer:
        optimizer.zero_grad()
        loss.mean().backward()
        pos_grad, dir_grad = input_pos.grad, input_dir.grad
        optimizer.step()

        ##### d. Log found gradients and predictions
        loss_trajectory.append(loss.detach().numpy())

        pos_grad_norm     = (np.linalg.norm(pos_grad), np.linalg.norm(dir_grad))
        gradients_norm.append(pos_grad_norm)

        perc_pred         = (prediction[0] + 1) / 2
        predictions.append(perc_pred)

        if verbose:
            parsing_bar.set_description(f"Gradient norm: loc {pos_grad_norm[0]:.4f}, dir {pos_grad_norm[1]:.4f}")

        ##### add location to trajectory
        scaled_loc = rescale_from_norm_params(input_pos.detach().numpy()[0], info_dict["xyz_centroid"], info_dict["xyz_max-min"])
        scaled_dir = rescale_from_norm_params(input_dir.detach().numpy()[0], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])

        actual_loc = np.hstack([scaled_loc, scaled_dir])
        #actual_loc = input_pos.detach().numpy()[0]
        trajectory.append(actual_loc)

        if loss.mean() < loss_threshold:
            break
    #final_residual = 0
    # print(sample_batch["output"].detach().numpy()[0], desired_target, predictions[-1])
    desired_percentages = (sample_batch["output"][0] + 1) / 2
    actual_percentages  = torch.Tensor(predictions[-1])
    # if MSE residual:
    final_residual = criterion(desired_percentages, actual_percentages).mean().detach().item()
    # if RMSE residual:
    final_residual = torch.sqrt(criterion(desired_percentages, actual_percentages)).mean().detach().item()
    
    #print(final_residual, desired_percentages, actual_percentages)
    performed_steps = i
    if debugging_return:
        return actual_loc, perc_pred, trajectory, gradients_norm, predictions, loss_trajectory, performed_steps, final_residual
    
    return actual_loc, perc_pred

def prediction_is_close_enough(desired_prediction, actual_prediction, admissible_interval, admissible_errors):
    '''
    Compare [0, 1] percentage predictions: 
    True -> desired_prediction is either close to 0, or close to actual prediction. 

    return True or False
    '''

    # debugging_dict["predictions"] - are the actual_percentages [0, 1] for the whole trajectory
    # desired_distribution - are percentages [0, 1]

    # Check if prediction is in +- distance from desired distribution
    # desired_prediction      = desired_distribution.detach().numpy() #* 2 - 1
    # actual_prediction       = np.array(debugging_dict["predictions"])[-1] #* 2 - 1
    
    # admissible_interval      = 0.1 # admisable interval in [-1, 1] tanh deviation / twice as it would be in percentages [0, 1]
    
    close_prediction      = np.logical_or(desired_prediction < admissible_interval\
                                        , np.abs(desired_prediction - actual_prediction) < admissible_interval)

    print("\ndesired_prediction:", np.round(desired_prediction, 2), "\nactual_prediction:", np.round(actual_prediction, 2)\
    , "\nclose_prediction:", close_prediction, "\n") 

    print("Not close prediction sum:", np.logical_not(close_prediction), np.logical_not(close_prediction).sum(), admissible_errors, np.logical_not(close_prediction).sum() <= admissible_errors)
    num_errors = np.logical_not(close_prediction).sum()
    if  num_errors <= admissible_errors:
        #print(f"Prediction  {i+1} is close enough:")
        return num_errors

    return -1


def interval_desired_target_loss(prediction, desired_target, intervals):
    '''Compute target if prediction can be within interval of desired_target:
    interval_target = interval_desired_target_loss(prediction, desired_target, intervals)
    '''
    #prediction and desired_target are in [-1, 1] (tanh)
    #Check if prediction in interval from target.
    pred_in_interval = np.logical_and(prediction < desired_target + intervals, 
                                    prediction > desired_target - intervals)

    #Exclude targets with label < .05 (5% = -0.9 in tanh):
    # pred_in_interval = np.logical_and(desired_target>-0.9, pred_in_interval)

    interval_target = torch.tensor(np.where(pred_in_interval, prediction, desired_target))
    
    return interval_target

def get_gradient_from_location_and_output(mock_location, mock_target, return_predictions=True):
    '''
        Deprecated - adapted to directly use a torch optimizer in the gradient_walk method above.
        Assemble initialize_trained_encoder and initialize_input_as_tensor to also return gradient from location to desired target.
        pos_grad, dir_grad, prediction = get_gradient_from_location_and_output(mock_location, mock_target, return_predictions=True)
    '''
    
    trained_encoder, info_dict = initialize_trained_encoder()

    sample_batch = initialize_input_as_tensor(mock_location, mock_target, info_dict)
    
    #mock_target     = [0.15875327587127686, 0.42250925302505493, 0.11981145292520523, 0.2989259660243988]
    mock_target     = torch.tensor([2*mt -1 for mt in mock_target]) #percentages target #not used for the moment

    # input_pos = sample_batch["input_pos_raw"]
    # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/2
    input_pos = torch.autograd.Variable(sample_batch["input_pos_raw"], requires_grad=True)
    input_dir = sample_batch["input_dir_raw"]


    #print("Input postion:", input_pos)
    #print("Input direction:", input_dir)
    output = trained_encoder(input_pos, input_dir, from_raw=True)
    #optimizer.zero_grad()
    prediction = (output.detach().numpy())

    # if gt_labels:
    labels    = sample_batch["output"]
    
    
    criterion       = torch.nn.MSELoss(reduction='none')
    loss            = criterion(output,labels)
    loss.mean().backward()

    #print("Predictions:", prediction)
    #print("\nDesired Output:", labels)

    #print("\nGradient to position:", input_pos.grad)
    #print("Gradient to direction:", input_dir.grad)
    pos_grad, dir_grad = input_pos.grad, input_dir.grad
    if return_predictions:
        return pos_grad, dir_grad, prediction



def initialize_input_as_tensor(mock_location, mock_target, info_dict, on_surface=None):
    '''
        return  a data loader from input mock_location and output mock_target, according to info_dict encoidng information
        sample_batch = initialize_input_as_tensor(mock_location, mock_target, info_dict)
        mock_location is (1, 6) list -> RAW location - "x", "y", "z", "xh", "yh", "zh".
    '''
    
    norm_params     = (info_dict["xyz_centroid"], info_dict["xyz_max-min"], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])
    
    test_df                = pd.DataFrame(mock_location, ["x", "y", "z", "xh", "yh", "zh"]).T
    test_df["f_xyz"]       = [mock_target]
    test_df["image_name"]  = "no_image_name"
    test_name              = "_".join(test_df[["x", "y", "z"]].astype(int).values[0].astype(str))

    test_path = f"./utils/assets/test_data/location_single_{test_name}.csv"

    info_dict["sli"] = np.sort(np.intersect1d(info_dict["classes_names"], info_dict["non_empty_classes_names"], return_indices=True)[1])
    # test_df = curr_neigborhood
    test_df.to_csv(test_path, index=False)
    # print(f"Info dict keys: {list(info_dict.keys())}")
    print(f"Selected Label indexes are: {list(zip(info_dict['non_empty_classes_names'], info_dict['sli']))}")
    ml = True # missing_labels / skip label normalization
    test_df, _, _   = process_locations_visibility_data_frame(test_path, norm_params, selected_label_indexes=info_dict["sli"], missing_labels=ml)

    print("test_df is:", test_df)

    #1.Data loader from points
    ml = False # not missing_labels / don't skip labels in the dataloader
    test_dl  = get_location_visibility_loaders(test_df, missing_labels=ml, only_test=True, batch_size=16, return_raw=True, on_surface=on_surface, norm_params=norm_params)
    # if on_surface is None:
    #     test_dl  = get_location_visibility_loaders(test_df, missing_labels=ml, only_test=True, batch_size=16, return_raw=True, on_surface=None, norm_params=None)
    # else:


    os.remove(test_path)
    sample_batch = test_dl.sampler.data_source[:1]
    # print("Sample batch is:", sample_batch)
    
    return sample_batch


import random
def set_seed(seed: int = 43) -> None:
    """
    Check:
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy#setting-up-random-seeds-in-pytorch
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")




############################################################
###### Draw queried locations experiment for exploration:###
############################################################
from utils.scripts.interest_heuristic_0    import get_o3d_pcd_from_coordinates
import open3d as o3d
def analyze_queried_locations(al_df=None, n_colors = 5, res_filter_threshold = .5, draw_o3d=True):
    """Invistigate queried locations Color Scene by label categories and label pallete: VIS 2024 figure 
    developed in iNeRF_adaption.ipynb"""
    vis_df, normp, _   = process_locations_visibility_data_frame("./utils/assets/test_data/locations_example.csv")

    
    targets_array = ((np.vstack(vis_df["f_xyz"])+1)/2)
    locs_array    = (vis_df.values)[:,:3]
    geometries    = []
    if "start_locs" in al_df:
        start_locs   = np.vstack(al_df["start_locs"].values)
        strat_colors = np.ones_like(start_locs) * .8
        start_pcd = get_o3d_pcd_from_coordinates(start_locs, strat_colors)
        geometries.append(start_pcd)

    #start_locs    = locs_array#np.vstack(al_df["start_locs"].values)[:,:3]#np.vstack([l[0] for l in locations])[:,:3]
    #start_targets = ((np.vstack(vis_df["f_xyz"])+1)/2)#np.array([t[0] for t in targets])
    
    if al_df is not None:
        achieved_locs = al_df.values[:,:3]#np.vstack([l[1] for l in locations])[:,:3]
        mean_losses   = np.array(al_df["residual"].values)
        scaled_residuals = mean_losses #/ mean_losses.max()
        filtered_ids  = np.where(scaled_residuals<res_filter_threshold)[0] #np.where(np.array(mean_losses)<.5)[0]
        #err_pallete = sns.color_palette("coolwarm", n_colors=n_colors)# Blue to Orange
        #err_pallete = sns.diverging_palette(125, 5, as_cmap=False, n=n_colors, s=60)# Greento Red
        err_pallete = sns.diverging_palette(260, 5, as_cmap=False, n=n_colors, s=60) # Blue to Red
        # loss_colors = np.array([err_pallete[::-1][int(ml*n_colors)-1] for ml in scaled_residuals])
        loss_colors = np.array([err_pallete[::-1][int(ml*n_colors)-1] for ml in mean_losses - mean_losses.min()])
        # loss_colors = np.array([err_pallete[int(ml*n_colors)-1] for ml in mean_losses - mean_losses.min()])

    ###Label colors pallete
    label_pallete        = sns.color_palette("tab10", n_colors=4) #[Buildings, Water, Trees, Sky]
    label_pallete[1]     = label_pallete[0]; label_pallete[0] = (0.549, 0.471, 0.318); label_pallete[3] = (1, 0.847, 0.012) #ffd803
    full_dominant_label  = np.argmax(targets_array > np.array(targets_array).mean(axis=0), axis=1)
    #start_dominant_label = np.argmax(start_targets > start_targets.mean(axis=0), axis=1)
    #start_label_colors = np.take(label_pallete, start_dominant_label, axis=0)
    full_locs = locs_array[:,:3][full_dominant_label!=3]
    full_label_colors  = np.take(label_pallete, full_dominant_label, axis=0)[full_dominant_label!=3]
    # full_label_colors = np.hstack([full_label_colors, np.ones((full_label_colors.shape[0], 1))*.2])

    #O3D point cloud assembly:
    # full_pcd      = get_o3d_pcd_from_coordinates(locs_array[:,:3], [0,1,0])
    full_pcd      = get_o3d_pcd_from_coordinates(full_locs, full_label_colors)
    # start_pcd     = get_o3d_pcd_from_coordinates(start_locs, [0,1,0])
    # start_pcd    = get_o3d_pcd_from_coordinates(start_locs, start_label_colors)
    geometries.append(full_pcd)# = [full_pcd]
    
    if al_df is not None:
        #adapted_pcd  = get_o3d_pcd_from_coordinates(achieved_locs, loss_colors)
        filtered_pcd = get_o3d_pcd_from_coordinates(achieved_locs[filtered_ids], loss_colors[filtered_ids])
        geometries.append(filtered_pcd)

    if draw_o3d:
        # o3d.visualization.draw_geometries([start_pcd])
        # o3d.visualization.draw_geometries([full_pcd])
        # o3d.visualization.draw_geometries([full_pcd, adapted_pcd])
        o3d.visualization.draw_geometries(geometries)

    return geometries#, achieved_locs, loss_colors, scaled_residuals, err_pallete, filtered_ids
