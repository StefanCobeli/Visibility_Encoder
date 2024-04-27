
from cgi import test
import numpy as np
import pandas as pd
import seaborn as sns

import time
import torch

import os


from utils.geometry_utils import *
from utils.test_location_encoder import parse_training_info

from utils.scripts.architectures.train_location_encoder import *
from utils.scripts.architectures.torch_nerf_src import network
# from utils.gradient_walk_utils import initialize_trained_encoder, intialize_input_as_tensor
# from utils.scripts.architectures.train_location_encoder import rescale_from_norm_params 



def query_locations(desired_distribution, num_locations=10, search_intervals=np.ones(4) * .2, lt=.01, seed=1):
    '''return num_locations with desired_distribution plus minus the search_intervals with a loss threshold smaller than lt'''
    
    np.random.seed(seed)
    
    vis_df, normp, _   = process_locations_visibility_data_frame("./utils/assets/test_data/locations_example.csv")  
    locs_array         = vis_df.values[:,:6].astype(float)
    targets_array = (np.vstack((vis_df["f_xyz"]))).tolist()
    
    num_ps      = [] #number performed_steps
    mean_losses = [] # final mean losses for each pair input target
    targets     = [] #(achieved, desired)
    locations   = [] #(start, achieved)
    spatial_differences = [] #(initial, final)

    n_trials  = num_locations
    max_steps = 100

    custom_target_d = desired_distribution #Distribution in percentages 
    desired_target  = (np.array(custom_target_d) * 2 - 1).tolist() #Distribution in tanh


    for i in tqdm(range(n_trials * 10)):
        
        crt_id = np.random.randint(0, len(locs_array))
        
        #des_id = np.random.randint(0, len(targets_array))
        #rd_ids += [crt_id, des_id]

        actual_loc    = locs_array[crt_id]
        actual_target = targets_array[crt_id]

            
        achieved_loc, perc_pred, tr, gn, prds, lstr, ps = gradient_walk(actual_loc, desired_target, search_intervals, max_steps, lt, True)

        if lstr[-1].mean() < lt:
            locations.append((actual_loc, achieved_loc))

            num_ps.append(ps)
            mean_losses.append(lstr[-1].mean())

            start    =  [(at+1)/2 for at in actual_target]
            achieved =  perc_pred
            desired  =  [(dt+1)/2 for dt in desired_target]
            targets.append([np.round(t, 2) for t in [achieved, desired]])
        
        if len(locations) >= num_locations:
            break
          
    #print(vis_df.columns.values)
    #print(vis_df.iloc[0])
    
    #print(pd.DataFrame(data=locations[0], columns=vis_df.columns.values[:6]))
    
    ach_locs = [l[1] for l in locations]
    al_df    = pd.DataFrame(data=ach_locs, columns=vis_df.columns.values[:6])#achived locations data frame
    
    al_df["f_xyz"] = [t[0] for t in targets]
    
    return al_df


def gradient_walk(actual_loc, desired_target, intervals=None, n_steps=10, loss_threshold=0, debugging_return=False, optimizer=None, verbose=False):
    '''
    Go from actual_loc to location looking like desired_target
    actual_loc     - x,y,z,zh,yh,zh;
    desired_target - [b,w,t,s] - according to the default model in - initialize_trained_encoder()
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
            parsing_bar.set_description(f"Gradient norm: loc {pos_grad_norm[0]:.4f}, dir {pos_grad_norm[0]:.4f}")

        ##### add location to trajectory
        scaled_loc = rescale_from_norm_params(input_pos.detach().numpy()[0], info_dict["xyz_centroid"], info_dict["xyz_max-min"])
        scaled_dir = rescale_from_norm_params(input_dir.detach().numpy()[0], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])

        actual_loc = np.hstack([scaled_loc, scaled_dir])
        #actual_loc = input_pos.detach().numpy()[0]
        trajectory.append(actual_loc)


        if loss.mean() < loss_threshold:
            break
    
    performed_steps = i
    if debugging_return:
        return actual_loc, perc_pred, trajectory, gradients_norm, predictions, loss_trajectory, performed_steps
    
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

def initialize_trained_encoder():
    '''
        Intialize encoder from hard coded file path and return also the training information in info_dict
        trained_encoder, info_dict = initialize_trained_encoder()
    '''
    
    mv = 350                     #model version
    mp = "./utils/assets/models/"# path to models folder
    info_dict       = parse_training_info(mp, mv)
    trained_encoder = network.nerf.NeRF(pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"]\
        ,  view_dir_dim=info_dict["enc_input_size"], feat_dim=256)
    trained_encoder.load_state_dict(torch.load(f"{mp}/encoder_{mv}.pt"))
    
    return trained_encoder, info_dict

def intialize_input_as_tensor(mock_location, mock_target, info_dict):
    '''
        return  a data loader from input mock_location and output mock_target, according to info_dict encoidng information
        sample_batch = intialize_input_as_tensor(mock_location, mock_target, info_dict)
    '''
    
    norm_params     = (info_dict["xyz_centroid"], info_dict["xyz_max-min"], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])
    
    test_df    = pd.DataFrame(mock_location, ["x", "y", "z", "xh", "yh", "zh"]).T
    test_df["f_xyz"] = [mock_target]
    test_df["image_name"] = "no_image_name"
    test_name  = "_".join(test_df[["x", "y", "z"]].astype(int).values[0].astype(str))

    test_path = f"./utils/assets/test_data/location_single_{test_name}.csv"

    # test_df = curr_neigborhood
    test_df.to_csv(test_path, index=False)
    ml = True # skip label normalization
    test_df, _, _   = process_locations_visibility_data_frame(test_path, norm_params, selected_label_indexes=info_dict["sli"], missing_labels=ml)

    #1.Data loader from points
    ml = False # don't skip labels in the dataloader
    test_dl  = get_location_visibility_loaders(test_df, missing_labels=ml, only_test=True, batch_size=16, return_raw=True)

    os.remove(test_path)
    sample_batch = test_dl.sampler.data_source[:1]
    
    return sample_batch