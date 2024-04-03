import numpy as np
import pandas as pd
import seaborn as sns
import shutil

import time
import torch

import os


from utils.geometry_utils import *

from utils.scripts.architectures.train_location_encoder import *
from utils.scripts.architectures.torch_nerf_src import network
from utils.test_location_encoder import parse_training_info


def reset_encoder_weights():
    '''
    Reset weights of model to the state before building removal.
    '''
    mp = "./utils/assets/models"# path to models folder
    mv = 350                     #model version
    
    backup_path = f"{mp}/encoder_{mv}_bk.pt"
    model_path  = f"{mp}/encoder_{mv}.pt"
    
    shutil.copyfile(backup_path, model_path)
    
    print(f"Model weights restored to original state in:\n\t{model_path}")



def remove_builiding_and_retrain_model():
    '''
    Train model on received locations with reassigned visibility values according to removed building.
    Save the retrained model in place of the original.
    '''
    
    mp              = "./utils/assets/models"# path to models folder
    mv              = 350                     #model version
    bs              = 2**10#11 and 9 work less well. 2**10 is the sweet spot
    rm_bs           = 32 # 26 worked well also, 16 and 64 worked worse
    num_epochs      = 2 #epochs for removal training
    epochs_factor   = 15 #times  num_epochs returns the training times over the entire dataset.

    start_time = time.time(); end_time=0



    print("Loading model!")
    info_dict       = parse_training_info(mp, mv)
    info_df         = pd.DataFrame({k:[info_dict[k]] for k in info_dict}).T
    trained_encoder = network.nerf.NeRF(pos_dim=info_dict["enc_input_size"], output_dim=info_dict["num_present_classes"],  view_dir_dim=info_dict["enc_input_size"], feat_dim=256)
    optimizer       = torch.optim.Adam(trained_encoder.parameters(), lr=1e-6, eps=1e-8)
    criterion       = torch.nn.MSELoss(reduction='none')

    trained_encoder.load_state_dict(torch.load(f"{mp}/encoder_{mv}.pt"))
    print("Loading data frames!")
    ########################################################################
    ################# 2. load training data ################################
    ########################################################################
    full_data_path                = "./utils/assets/test_data/locations_example.csv"
    norm_params                   = (info_dict["xyz_centroid"], info_dict["xyz_max-min"], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])
    visibility_dataset_df, _, _   = process_locations_visibility_data_frame(full_data_path, norm_params, selected_label_indexes=info_dict["sli"] )
    _, _, train_df, test_df       = get_location_visibility_loaders(processed_vis_loc_df=visibility_dataset_df, test_size=0.2, batch_size=bs, pos_enc_dim=info_dict["pos_enc_dim"], seed=1, return_dfs=True)
    # print("Loaded data frames!")

    ########################################################################
    ################# 3. load removed data ################################
    ########################################################################
    test_loc_path    = "./utils/assets/removed_buildings/removedBuilding.csv"
    removed_df, _, _ = process_locations_visibility_data_frame(test_loc_path, norm_params, selected_label_indexes=info_dict["sli"], missing_labels=False)
    removed_train_df = pd.merge(train_df, removed_df,  how='inner', left_on=['x','y',"z", 'xh', 'yh', 'zh'], right_on = ['x','y',"z", 'xh', 'yh', 'zh'])
    removed_test_df  = pd.merge(test_df, removed_df,  how='inner', left_on=['x','y',"z", 'xh', 'yh', 'zh'], right_on = ['x','y',"z", 'xh', 'yh', 'zh'])


    print(f"{removed_train_df.shape[0]} recomputed locations from the training set.")
    print(f"{removed_test_df.shape[0]} recomputed locations from the test set.")

    ##########################################################################
    ################# 4. replace values of f_xyz if they appear in the join of removed buildings:
    #a. apply if image_name in removed than assign to row f_xyz_y of the removed, oterwise keeep f_xyz of original.
    ##########################################################################
    reassigned_train_dict = {k:v for (k, v )in removed_train_df[["image_name_x", "f_xyz_y"]].values}
    reassigned_test_dict  = {k:v for (k, v )in removed_test_df[["image_name_x", "f_xyz_y"]].values}

    train_df["f_xyz"] = train_df.apply(lambda r: reassigned_train_dict[r["image_name"]] \
                   if r["image_name"] in reassigned_train_dict else r["f_xyz"], axis=1)
    test_df["f_xyz"] = test_df.apply(lambda r: reassigned_test_dict[r["image_name"]] \
                   if r["image_name"] in reassigned_test_dict else r["f_xyz"], axis=1)

    removed_train_df = train_df[train_df["image_name"].isin(removed_train_df["image_name_x"])]
    removed_test_df  = test_df[test_df["image_name"].isin(removed_test_df["image_name_x"])]

    # #4b. Create new loaders from new reassigned scalar field values:
    print("\nCreating corresponding new train and test dataloaders...")
    train_loader, train_df = get_location_visibility_loaders(train_df, batch_size=bs, only_train=True, pos_enc_dim=info_dict["pos_enc_dim"], seed=1, return_dfs=True)
    test_loader, test_df   = get_location_visibility_loaders(test_df, batch_size=bs, only_test=True, pos_enc_dim=info_dict["pos_enc_dim"], seed=1, return_dfs=True)
    rm_test_loader         = get_location_visibility_loaders(removed_test_df, batch_size=rm_bs, only_test=True, pos_enc_dim=info_dict["pos_enc_dim"], seed=1, return_dfs=False)
    rm_train_loader        = get_location_visibility_loaders(removed_train_df, batch_size=rm_bs, only_test=True, pos_enc_dim=info_dict["pos_enc_dim"], seed=1, return_dfs=False)


    # train_df.shape, test_df.shape
    #######################################################################
    ################  5. Fine-tuning loop: ################################
    #######################################################################

    training_progress   = tqdm(range(int(num_epochs * epochs_factor)))
    logging_steps       = 1#2#5
    tr_losses_history   = [info_dict["training_losses_history"][-1]]
    test_losses_history = [info_dict["test_losses_history"][-1]]
    rm_losses_history   = [info_dict["test_losses_history"][-1]]

    for epoch in training_progress:

        #Train on entire dataset
        tr_loss = run_one_epoch_location_encoder(trained_encoder, criterion, optimizer\
                                       , train_loader, training_epoch=True) 

        #Train multiple times on newly reassigned locations: 
        for i in range(num_epochs):
            _ = run_one_epoch_location_encoder(trained_encoder, criterion, optimizer\
                                           , rm_train_loader, training_epoch=True)
        tr_losses_history.append(tr_loss) 

        if epoch % logging_steps == 0:
            # Entire Test
            test_loss = run_one_epoch_location_encoder(trained_encoder, criterion, optimizer\
                                       , test_loader, training_epoch=False)
            test_losses_history.append(test_loss)

            # Only Removed region test:
            rm_test_loss = run_one_epoch_location_encoder(trained_encoder, criterion, optimizer\
                                       , rm_test_loader, training_epoch=False)
            rm_losses_history.append(rm_test_loss)

            if rm_test_loss.mean() < test_loss.mean() and end_time==0:
                end_time = time.time() 
                print(f"Removed building loss caught test loss after {epoch+1} ({(epoch+1)*num_epochs}) epochs, in")
                print(f"\t{end_time - start_time:.2f} seconds.")

            if test_losses_history[0].mean() > test_losses_history[-1].mean() and end_time != 0:
                end_time = time.time() 
                print(f"Test loss got back to original performance after {epoch+1} ({(epoch+1)*num_epochs}) epochs, in")
                print(f"\t{end_time - start_time:.2f} seconds.")
                break

        training_progress.set_description(f'Epoch {epoch +1} / {num_epochs* epochs_factor}'+
              f'- Training loss {tr_loss.mean()**.5:.5f} - test loss {test_loss.mean()**.5:.5f}'  
             f" - removed test loss {rm_test_loss.mean()**.5:.5f}")

    end_time = time.time() 

    ##########################################################################
    ##########################################################################
    ##########################################################################
    print(f"Fine tuning for {num_epochs} on the full and {num_epochs*num_epochs} on removed buiding took:")
    print(f"\t{end_time - start_time:.2f} seconds.")
    
    
    torch.save(trained_encoder.state_dict(), f"{mp}/encoder_{mv}.pt")

    print(f"\tModel with removed buidiling saved in place of the original:\n\t{mp}/encoder_{mv}.pt.")
    #trained_encoder.load_state_dict(torch.load(f"{mp}/encoder_{mv}.pt"))

    return test_losses_history, rm_losses_history