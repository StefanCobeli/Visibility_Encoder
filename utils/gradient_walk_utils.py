
from cgi import test
import numpy as np
import pandas as pd
import seaborn as sns

import time
import torch

import os


from utils.geometry_utils import *
from utils.geometry_utils import surface_parametric
from utils.test_location_encoder import parse_training_info

from utils.scripts.architectures.train_location_encoder import *
from utils.scripts.architectures.torch_nerf_src import network
# from utils.gradient_walk_utils import initialize_trained_encoder, intialize_input_as_tensor
# from utils.scripts.architectures.train_location_encoder import rescale_from_norm_params 


def choose_model_based_on_query(desired_distribution):
    """
    Find what model should be used.
    Return info_dict with model_path entry and 
    rectified query as list of length being the output of the model.
    
    rectified_query: -1 where nothing specified and value where specified.

    return info_dict, rectified_query
    """
    query_labels = None
    #Scenarios:
    #Semantics
    #1.  ['building', ' water', 'tree', 'sky']
    semantics_pricipal   = ['building', 'water', 'tree', 'sky'] 
    labels, label_ids, query_ids = np.intersect1d(semantics_pricipal, list(desired_distribution.keys()), return_indices=True)
    if len(labels) > 0: #semantics_pricipal case
        info_dict_path = "./utils/assets/data/splits_physical/models/training_info_100.json"
        rectified_distribution = np.zeros_like(semantics_pricipal, dtype=float) - 1
        rectified_distribution[label_ids] = [desired_distribution[semantics_pricipal[qi]] for qi in label_ids]

        print(labels, label_ids, query_ids)
        print(f"Querying for principal semantics:")
        query_labels = semantics_pricipal

    #2.  [' building' ' water' ' road ' ' sidewalk' ' surface' ' tree' ' sky']
    semantics_adders     = ['road', 'sidewalk', 'surface']
    semantics_full       = ['building', 'water', 'road', 'sidewalk', 'surface', 'tree', 'sky']
    labels, label_ids, query_ids = np.intersect1d(semantics_adders, list(desired_distribution.keys()), return_indices=True)
    if len(labels) > 0: #semantics_full case
        labels, label_ids, query_ids = np.intersect1d(semantics_full, list(desired_distribution.keys()), return_indices=True)
        info_dict_path = "./utils/assets/data/full_semantics/models/training_info_1000.json"
        rectified_distribution = np.zeros_like(semantics_full, dtype=float)  - 1
        rectified_distribution[label_ids] = [desired_distribution[semantics_full[qi]] for qi in label_ids]
        print("\nFull semantics found:")
        print(f"Querying for full semantics: \n\t{semantics_full}")
        query_labels = semantics_full

    #Perception:
    #3. ["greeness", "openness", "imageability", "encolusre", "walkability", "serenity"]
    perceptions          = ["greenness", "openness", "imageability", "enclosure", "walkability", "serenity"]
    labels, label_ids, query_ids = np.intersect1d(perceptions, list(desired_distribution.keys()), return_indices=True)
    if len(labels) > 0: #perception case
        info_dict_path = "./utils/assets/data/perception_metrics/models/training_info_350.json"
        rectified_distribution = np.zeros_like(perceptions, dtype=float) - 1
        rectified_distribution[label_ids] = [desired_distribution[perceptions[qi]] for qi in label_ids]
        print(f"Querying for perception semantics: \n\t{perceptions}")
        query_labels = perceptions
    
    print(f"The query was:\n\t{desired_distribution}")
    print(f"Looking for the rectified query:")
    print(f"\t{dict(zip(query_labels, rectified_distribution))}")
    print("~The rectified query will be further tanh normalized (-1, 1) in the nerfs.EncoderNeRFSDataset constructor.\n")

    #Building Materials:
    #4. ---
    #Combinations of any values - semantics, perception and buildings:


    # model_folder = "/".join(trained_model_path.split("/")[:-1])
    # model_name   = trained_model_path.split("/")[-1].replace("training_info", "encoder").replace("json", "pt")
     # trained_encoder.load_state_dict(info_dict["model_path"])

    #print(info_dict_path)
    info_dict = parse_training_info(info_dict_path=info_dict_path)
    info_dict["model_path"] = info_dict_path.replace("training_info", "encoder").replace("json", "pt")
    return info_dict, rectified_distribution # to be passed to gradient_walk_on_surface -  used for norm params and model path.

def initialize_trained_encoder(encoder_name="semantics"):
    '''
        Intialize encoder from hard coded file path and return also the training information in info_dict
        trained_encoder, info_dict = initialize_trained_encoder()
        encoder_name - {"semantics", "perception", "buildings", "general"}
    '''
    
    mv = 350   #350 - trained with 4 sematic classes                  #model version
    # mv = 1000 #1000 - trained with 7 sematic classes and 10 depth estimators
    mp = "./utils/assets/models/"# path to models folder
    info_dict       = parse_training_info(mp, mv)
    trained_encoder = network.nerf.NeRF(pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"]\
        ,  view_dir_dim=info_dict["enc_input_size"], feat_dim=256)
    trained_encoder.load_state_dict(torch.load(f"{mp}/encoder_{mv}.pt"))
    
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
    
    if type(desired_distribution) is dict: 
        #Make search intervals based on the length of desired_distribution
        info_dict, rectified_distribution = choose_model_based_on_query(desired_distribution)
        interval =  search_intervals[0]
        search_intervals = np.where(rectified_distribution==-1, 10, np.ones_like(rectified_distribution) * interval)

        desired_distribution = torch.tensor(rectified_distribution).to(torch.float32)

        print(f"Search Intervals:\n\t{search_intervals}")

    if type(desired_distribution) is list:
        desired_distribution = torch.tensor([float(x) for x in desired_distribution]).to(torch.float32)
        info_dict = None
    #print(desired_distribution)
    
    n_trials  = num_locations * 10
    #Needed just for view_dir intialization
    print("\nProcessing locations_example data for untilted viewing directions. Locations are sampled randomly.")
    vis_df, normp, _   = process_locations_visibility_data_frame("./utils/assets/test_data/locations_example.csv")  
    locs_array         = vis_df.values[:,:6].astype(float)
    
    ach_locs    = []
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
        
        view_dir           = torch.tensor(locs_array[np.random.randint(locs_array.shape[0])][3:], requires_grad=False)#, requires_grad=True) # change view_dir to random picking
        a, b               = torch.tensor(np.random.random()), torch.tensor(np.random.random()) #torch.rand(2) 
        parameters         = (a, b)
        # print(a, b)
        
        raw_pos, debugging_dict = gradient_walk_on_surface(parameters, view_dir, desired_distribution, surface_basis, surface_type\
                 , intervals=search_intervals, n_steps=max_steps, loss_threshold=lt, lrate=lrate, debugging_return=True, verbose=False\
                 , info_dict=info_dict)
        
        ach_locs.append(np.hstack([raw_pos, debugging_dict["last_view_dir"]]))
        debug_dicts.append(debugging_dict)
        
        if len(ach_locs) >= num_locations:
            break 
        
    al_df               = pd.DataFrame(data=ach_locs, columns=vis_df.columns.values[:6])#achived locations data frame
    

    # print(list(info_dict.keys()))
    # print(info_dict["non_empty_classes_names"])
    if info_dict is None:
        al_df["f_xyz"]      = [d["predictions"][-1] for d in debug_dicts] # Originally returned as list of values
    else:
        al_df["f_xyz"]      = [dict(zip(info_dict["non_empty_classes_names"], d["predictions"][-1])) for d in debug_dicts] # Return as dictionary of values.
    al_df["residual"]   = [d["final_residual"] for d in debug_dicts]
    al_df["steps"]      = [len(d["trajectory"]) for d in debug_dicts]
    al_df["start_locs"] = [d["trajectory"][0] for d in debug_dicts]
    al_df["start_views"] = [d["trajectory_view_dir"][0] for d in debug_dicts]
    

    return al_df.sort_values("residual")


def gradient_walk_on_surface(parameters, view_dir, desired_target, surface_basis, surface_type, intervals=np.ones(4) * .1, n_steps=10, loss_threshold=0.1, lrate=5*1e-2, debugging_return=True, verbose=True, info_dict=None):
    ''' 
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

    norm_params                = (torch.tensor(info_dict["xyz_centroid"]), torch.tensor(info_dict["xyz_max-min"]), torch.tensor(info_dict["xyzh_centroid"]), torch.tensor(info_dict["xyzh_max-min"]))
    trained_encoder            = network.nerfs.NeRFS(p, c, r, norm_params, surface_type=surface_type, pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"],  view_dir_dim=info_dict["enc_input_size"])
    
    trained_encoder.load_state_dict(torch.load(info_dict["model_path"]))
    # trained_encoder.load_state_dict(info_dict["model_path"])
    # trained_encoder.load_state_dict(torch.load(f"./utils/assets/models/encoder_1000.pt"))
    criterion                  = torch.nn.MSELoss(reduction='none')
    
    #0. Load data to torch dataset
    #TODO: remove pcr from dataset condructor and surface_type
    # print("desired_target:", desired_target)
    param_ds     = network.nerfs.EncoderNeRFSDataset(init_a, init_b, p, c, r, view_dir, desired_target, "square", norm_params)
    sample_batch = param_ds[0]
    input_a, input_b, input_dir = sample_batch["a"], sample_batch["b"], sample_batch["view_dir"]
    # print(init_a, init_b)
    # lrate      = 5*1e-2
    optimizer  = torch.optim.Adam(params=[input_a, input_b, view_dir], lr=lrate)
    
    gradients_norm  = []; predictions = []; trajectory = []; loss_trajectory = []; inputs = []; trajectory_view_dir = []#maybe trajectory view if needed
    if verbose:
        parsing_bar     = tqdm(range(n_steps))
    else:
        parsing_bar = range(n_steps)
    
    print()
    for i in parsing_bar:
        # print(f"Optimization step {i}/{n_steps}")

        inputs.append((input_a.detach().numpy().tolist(), input_b.detach().numpy().tolist(), input_dir.detach().numpy().tolist()))
        ##### a. Predict output distribution
        raw_pos, raw_view, output     = trained_encoder(input_a, input_b, input_dir)
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
            print(f"Jumped of the surface on step {i}/{n_steps}, with a: {input_a.item()} and b: {input_b.item() }")
            break

        ##### d. Log found gradients and predictions as percentages
        loss_trajectory.append(loss.detach().numpy())

        pos_grad_norm     = (np.linalg.norm(a_grad), np.linalg.norm(b_grad))#, np.linalg.norm(dir_grad))
        gradients_norm.append(pos_grad_norm)
        
        if verbose:
            parsing_bar.set_description(f"Gradient norm: a {pos_grad_norm[0]:.4f}, b {pos_grad_norm[1]:.4f}, dir {pos_grad_norm[2]:.4f}")

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
        debugging_dict = {"final_residual":final_residual, "trajectory":trajectory, "last_view_dir":raw_view.detach().numpy(), "predictions":predictions, "gradients_norm":gradients_norm, "loss_trajectory":loss_trajectory, "inputs":inputs, "trajectory_view_dir":trajectory_view_dir}
        return raw_pos, debugging_dict
    else:
        return raw_pos, perc_pred


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


    for i in tqdm(range(n_trials * 10)):
        
        crt_id = np.random.randint(0, len(locs_array))
        
        #des_id = np.random.randint(0, len(targets_array))
        #rd_ids += [crt_id, des_id]

        actual_loc    = locs_array[crt_id]
        actual_target = targets_array[crt_id]

        ### Gradient Walk: - For each location until finding num_locations.
        # print("Intervals:", search_intervals)
        achieved_loc, perc_pred, tr, gn, prds, lstr, ps, fr = gradient_walk(actual_loc, desired_target, search_intervals, max_steps, lt, True)

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

    sample_batch    = intialize_input_as_tensor(actual_loc, desired_target, info_dict)

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
        optimizer  = torch.optim.Adam(params=[input_pos, input_dir], lr=lrate)

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


def interval_desired_target_loss(prediction, desired_target, intervals):
    '''Compute target if prediction can be within interval of desired_target:
    interval_target = interval_desired_target_loss(prediction, desired_target, intervals)
    '''
    pred_in_interval = np.logical_and(prediction < desired_target + intervals, 
                                    prediction > desired_target - intervals)

    interval_target = torch.tensor(np.where(pred_in_interval, prediction, desired_target))
    
    return interval_target

def get_gradient_from_location_and_output(mock_location, mock_target, return_predictions=True):
    '''
        Deprecated - adapted to directly use a torch optimizer in the gradient_walk method above.
        Assemble initialize_trained_encoder and intialize_input_as_tensor to also return gradient from location to desired target.
        pos_grad, dir_grad, prediction = get_gradient_from_location_and_output(mock_location, mock_target, return_predictions=True)
    '''
    
    trained_encoder, info_dict = initialize_trained_encoder()

    sample_batch = intialize_input_as_tensor(mock_location, mock_target, info_dict)
    
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



def intialize_input_as_tensor(mock_location, mock_target, info_dict, on_surface=None):
    '''
        return  a data loader from input mock_location and output mock_target, according to info_dict encoidng information
        sample_batch = intialize_input_as_tensor(mock_location, mock_target, info_dict)
        mock_location is (1, 6) list -> RAW location - "x", "y", "z", "xh", "yh", "zh".
    '''
    
    norm_params     = (info_dict["xyz_centroid"], info_dict["xyz_max-min"], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])
    
    test_df                = pd.DataFrame(mock_location, ["x", "y", "z", "xh", "yh", "zh"]).T
    test_df["f_xyz"]       = [mock_target]
    test_df["image_name"]  = "no_image_name"
    test_name              = "_".join(test_df[["x", "y", "z"]].astype(int).values[0].astype(str))

    test_path = f"./utils/assets/test_data/location_single_{test_name}.csv"

    # test_df = curr_neigborhood
    test_df.to_csv(test_path, index=False)
    ml = True # missing_labels / skip label normalization
    test_df, _, _   = process_locations_visibility_data_frame(test_path, norm_params, selected_label_indexes=info_dict["sli"], missing_labels=ml)


    #1.Data loader from points
    ml = False # not missing_labels / don't skip labels in the dataloader
    test_dl  = get_location_visibility_loaders(test_df, missing_labels=ml, only_test=True, batch_size=16, return_raw=True, on_surface=on_surface, norm_params=norm_params)
    # if on_surface is None:
    #     test_dl  = get_location_visibility_loaders(test_df, missing_labels=ml, only_test=True, batch_size=16, return_raw=True, on_surface=None, norm_params=None)
    # else:


    os.remove(test_path)
    sample_batch = test_dl.sampler.data_source[:1]
    
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
