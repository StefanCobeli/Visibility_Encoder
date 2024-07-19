import numpy as np
import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from utils.scripts.architectures.torch_nerf_src import network
# from tqdm.notebook import tqdm
from tqdm import tqdm



def run_one_epoch_location_encoder(encoder_net, criterion, optimizer, data_loader, training_epoch=True, return_predictions=False, gt_labels=True):
    '''
    Only one epoch through the dataloader.
    '''
    #sum of losses for each sample to be normalized by len(dataset)
    current_losses = np.zeros(encoder_net.fc_out.out_features)
    predictions    = []
    all_losses     = [] if gt_labels else [current_losses]
    #print(len(data_loader.dataset))

    for i, sample_batch in enumerate(data_loader):
        
        input_pos = sample_batch["input_pos"]
        input_dir = sample_batch["input_dir"]
            
        output = encoder_net(input_pos, input_dir)
        optimizer.zero_grad()
        predictions.append(output.detach().numpy())

        if gt_labels:
            labels    = sample_batch["output"]
            loss      = criterion(output,labels)
            
            if training_epoch:
                loss.mean().backward()
                optimizer.step()
            current_losses += loss.sum(axis=0).detach().numpy() 
            all_losses.append(loss.detach().numpy())

    mean_epoch_loss = (current_losses / len(data_loader.dataset))

    if return_predictions :
        return mean_epoch_loss, np.vstack(all_losses), np.vstack(predictions)#,np.array(predictions)
    #print(len(data_loader.dataset), return_predictions, predictions)
    return mean_epoch_loss#, current_losses


def get_location_visibility_encoder(enc_input_size, num_present_classes, feat_dim=256):
    '''
    Setup training model: 
        enc_input_size: stands for both the size of the location and of the direction.
        num_present_classes: size of the output layer.
    returns: encoder_net, criterion, optimizer, scheduler
    '''
    #Hyperparameters:
    lr_start            = 1e-5 
    
    #Model and loss declaration
    
    encoder_net = network.nerf.NeRF(pos_dim=enc_input_size, output_dim=num_present_classes, view_dir_dim=enc_input_size, feat_dim=feat_dim) 
    criterion   = torch.nn.MSELoss(reduction='none')
    optimizer   = torch.optim.Adam(encoder_net.parameters(), lr=lr_start, eps=1e-8)#,weight_decay=
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    return encoder_net, criterion, optimizer, scheduler


def rescale_from_norm_params(point, mean, dev):
    '''
    Inverse of normalize_visibility_dataframe method from train_location_encoder.
    when norm_columns is passed and where norm_columns values are taken from:
    (info_dict["xyz_centroid"], info_dict["xyz_max-min"], info_dict["xyzh_centroid"], info_dict["xyzh_max-min"])

    Example:
    rescale_from_norm_params(trajectory[0][:3], info_dict["xyz_centroid"], info_dict["xyz_max-min"])

    return scaled_point
    '''
    scaled_point = (point * dev) + mean
    
    return scaled_point


def normalize_visibility_dataframe(vis_df, norm_columns, train_mean_dev=None):
    '''
    vis_df normalized on norm_columns
    Mean normalization:
        https://en.wikipedia.org/wiki/Feature_scaling#Methods
        
    train_mean_dev - if not none it should be tuple of (mean, std) where mean is (x,y,z) and std is scalar
    '''
    
    if train_mean_dev is None:
        vd_mean   = vis_df[norm_columns].mean().values
        vd_max    = vis_df[norm_columns].max().values
        vd_min    = vis_df[norm_columns].min().values
        vd_dev    = (vd_max - vd_min).max()
    else:
        vd_mean, vd_dev = train_mean_dev
        
    new_names         = [n + "n" for n in norm_columns]
    vis_df[new_names] = (vis_df[norm_columns] - vd_mean) / vd_dev
    
    return vis_df, vd_mean, vd_dev

def normalize_visibility_labels(vis_df, train_mean_dev=None):
    '''
    vis_df to be normalized on "f_xyz" column - independent feature-wise normalization
    Mean normalization:
        https://en.wikipedia.org/wiki/Feature_scaling#Methods
        
    train_mean_dev - (mean, std) where mean and std are for each feature
    '''
    label_matrix = np.vstack(vis_df["f_xyz"])
    if train_mean_dev is None:
        label_mean   = label_matrix.mean(axis=0)
        label_max    = label_matrix.max(axis=0)
        label_min    = label_matrix.min(axis=0)
        label_dev    = (label_max - label_min)#.max()
    else:
        label_mean, label_dev = train_mean_dev
    
    vis_df["f_xyz_prenorm"] = vis_df["f_xyz"]
    vis_df["f_xyz"]         = (label_matrix - label_mean) / label_dev
    
    return vis_df, label_mean, label_dev

def process_locations_visibility_data_frame(file_store, norm_params=None, min_percentage_per_class=.1, label_split=",", missing_labels=False, selected_label_indexes=[0,1,5,6]):
    '''
    Process locations.csv file from file_store. 
    Table expected columns:
        - x,y,z,xh,yh,zh,f_xyz
    Normalize coordinates and only consider non empty columns as labels.
    min_percentage_per_class - minimum percentage of a class to not be considered empty.
    Returns:
    processed_data_frame and indexes of non empty labels
    normalization_paramerers - (xyz_mean, xyz_dev, xyzh_mean, xyzh_dev)
    non_empty_classes        - array with true, false entries based on which f_xyz was kept.
    return vis_df_n, norm_params, non_empty_classes
    '''
    # print("Processing")
    
    vis_df                                  = pd.read_csv(file_store)
    # vis_df[['x','y',"z", 'xh', 'yh', 'zh']] = vis_df[['x','y',"z", 'xh', 'yh', 'zh']].round(3)
    
    if norm_params is None:
        vis_df_n, xyz_mean, xyz_dev   = normalize_visibility_dataframe(vis_df, ["x", "y", "z"])
        vis_df_n, xyzh_mean, xyzh_dev = normalize_visibility_dataframe(vis_df_n, ["xh", "yh", "zh"])
        norm_params = (xyz_mean, xyz_dev, xyzh_mean, xyzh_dev)
    else:
        xyz_mean, xyz_dev, xyzh_mean, xyzh_dev = norm_params
        vis_df_n, _, _  = normalize_visibility_dataframe(vis_df, ["x", "y", "z"], train_mean_dev=(xyz_mean, xyz_dev))
        vis_df_n, _, _  = normalize_visibility_dataframe(vis_df_n, ["xh", "yh", "zh"], train_mean_dev=(xyzh_mean, xyzh_dev))
    
    if missing_labels:
        return vis_df_n, norm_params, None
    
    if label_split is None:
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').replace("\n", "").split(" ") if d.isdigit()])
    if label_split == ",":
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').split(",") if d.isdigit()])

    #Normalize Labels by the sum of each row. Predictions will be adding up to 1
    max_row_value                      = sum(vis_df["f_xyz_raw"].iloc[0])

    # Filter labels either by index or by occurence
    if selected_label_indexes is None:
        #a. Filter down to only labels that appear:
        minimum_occurances                 = min_percentage_per_class * max_row_value * vis_df_n.shape[0]
        indvidual_class_occurences         = np.sum(np.vstack((vis_df_n["f_xyz_raw"].values)), axis=0)
        non_empty_classes                  = indvidual_class_occurences > minimum_occurances
        # print(selected_label_indexes, non_empty_classes)
    else:
        #b. Filter by selected indexes.
        non_empty_classes = np.in1d(np.arange(len(vis_df_n["f_xyz_raw"].iloc[0])), selected_label_indexes)
        # print(non_empty_classes)

    #Keep only labels satisfying condition (occurences or selection)
    vis_df_n["f_xyz"]     = vis_df["f_xyz_raw"].apply(lambda d: \
                                             [p for (p, e) in zip(d, non_empty_classes) if e])
    #Normalize appearances and strech them between -1, +1:
    vis_df_n["f_xyz"]     = vis_df["f_xyz"].apply(lambda d: [(2 * x) / max_row_value - 1 for x in d]) 

    return vis_df_n, norm_params, non_empty_classes

def get_location_visibility_loaders(processed_vis_loc_df, train_set_percentage=1, test_size=0.2, batch_size=32, pos_enc_dim=10, seed=1, only_train=False, only_test=False, missing_labels=False, return_dfs=False, return_raw=False, on_surface=None, norm_params=None):
    """Return train and test loaders based on processed visibility data frame
    pos_enc_dim  # 4 or 10 #See NeRF paper section 5.1 Positional encoding, page 8 - L = 4 or L=10 for γ(d).
    only_train or only_test  - forces the return of only one loader and dataframe without random seed split. 
    return_raw - dataloader also containing original location and angles with linked gradients.
    return train_loader, test_loader, train_df, test_df (if return_dfs is True)
    return train_loader, test_loader
    """
    if only_test or only_train:
        loc_df = processed_vis_loc_df
    else:
        np.random.seed(seed)
        train_df, test_df     = train_test_split(processed_vis_loc_df, test_size=test_size)

    label_column_name     = "f_xyz"
    features_column_names = ["xn", "yn", "zn", "xhn", "yhn", "zhn"]

    #a. Enitre dataframe is passed to a single loader (if split is done before dataframe passed to the method).
    if only_test or only_train:
        loc_dataset     = network.nerf.EncoderNeRFDataset(loc_df, label_column_name=label_column_name, features_column_names=features_column_names, pos_enc_dim=pos_enc_dim, missing_labels=missing_labels, return_raw=return_raw, on_surface=on_surface, norm_params=norm_params)
        loc_loader      = torch.utils.data.DataLoader(dataset=loc_dataset, batch_size=batch_size)#, shuffle=True) # load data
        if return_dfs:
            return loc_loader, loc_df 
        else: 
            return loc_loader
        #return loc_loader, loc_df if return_dfs else loc_loader
    
    #b. dataframe is split in test and train loaders
    train_size       = int(train_df.shape[0] * train_set_percentage)
    train_df_subset  = train_df[:train_size]

    training_dataset = network.nerf.EncoderNeRFDataset(train_df_subset, label_column_name=label_column_name, features_column_names=features_column_names, pos_enc_dim=pos_enc_dim, return_raw=return_raw, on_surface=on_surface, norm_params=norm_params)
    testing_dataset  = network.nerf.EncoderNeRFDataset(test_df, label_column_name=label_column_name, features_column_names=features_column_names, pos_enc_dim=pos_enc_dim, return_raw=return_raw, on_surface=on_surface, norm_params=norm_params)

    train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True) # load data
    test_loader  = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True) # load data
    if return_dfs:
        return train_loader, test_loader, train_df, test_df
    else:
        return train_loader, test_loader
    
    # return train_loader, test_loader, train_df, test_df if return_dfs else train_loader, test_loader


def train_model_on_data(data_path, num_epochs=200, tsp=1, mpc = .05, separate_test=False, selected_label_indexes=None, file_name="locations", return_info=False):
    '''
    mpc: min_percentage_per_class - 0-1 how often a class should appear in the set to be considered - default .05.
    tsp: training set percentage  - out of the 80% of the data on how many samples should the model be trained - default 1.

    returns: encoder_net, trlh, tlh, visibility_dataset_df training_info_df (training_info_df only if return_info)
    '''

    import time

    st = time.time()

    NY_MESH = True
    # NY_MESH = False
    file_store = data_path + f"/{file_name}.csv"
    if separate_test:
        file_store_train = data_path + "/locations_train.csv"
        file_store_test  = data_path + "/locations_test.csv"
    
    if NY_MESH:

        ins_dict = {c["color"] : c["class"] for c in  \
                pd.read_csv(f"{data_path}/classes_index.csv"\
                            , index_col=False).to_dict(orient="records")
                   }
        label_split=","
        classes_names = list(ins_dict.values())
    else:
        from utils.scripts.helper_ply_SensatUrban_0 import ins_dict
        label_split=None
        file_store = "./data/cambridge_block_8_fragment_semantic_locs-2500_dirs-5_visual/locations.csv"
        classes_names = list(ins_dict.values())+["empty"]


    #1. Read locations.csv and process data frame with normalized coordinate inputs and labels
    visibility_dataset_df, norm_params, nec = process_locations_visibility_data_frame(file_store\
                                             , min_percentage_per_class=mpc, label_split=label_split, selected_label_indexes=selected_label_indexes )
    
    #If new feature scaling, otherwise comment also the saving of norm_params[4] and norm_params[5] 
    # visibility_dataset_df, norm_params, nec = process_locations_visibility_data_frame_with_labels(file_store\
    #                                          , min_percentage_per_class=mpc, label_split=label_split, selected_label_indexes=selected_label_indexes )
    #a. Change to train and test df split
    #b. create normalization method. Train df should return also normalization constants
    #c. Change point 2. - train and test Loaders should be returned indiviudally.

    non_empty_classes_names                  = np.array(classes_names)[nec]
    print(f"Found {nec.sum()} classes appearing more than {mpc*100:.1f}%: {non_empty_classes_names}")


    trlh = []
    tlh  = []

    #2. setup `torch` dataset and loaders
    pos_enc_dim = 10
    train_loader, test_loader = get_location_visibility_loaders(processed_vis_loc_df=visibility_dataset_df\
    , train_set_percentage=tsp, test_size=0.2, batch_size=32, pos_enc_dim=pos_enc_dim, seed=1)

    #3. Setup NeRF-like location encoder:
    enc_input_size, num_present_classes          = train_loader.dataset.input_dir.shape[1], train_loader.dataset.output.shape[1]
    encoder_net, criterion, optimizer, scheduler =  get_location_visibility_encoder(enc_input_size, num_present_classes)

    #4. Training loop
    print(f"Training data percentace {100 * tsp:.2f}% - {int(tsp * len(visibility_dataset_df)*.8):,} samples, for {num_epochs} epochs:")
    training_progress   = tqdm(range(num_epochs))
    logging_steps       = 2#5
    tr_losses_history   = []
    test_losses_history = []
    for epoch in training_progress:

        tr_loss = run_one_epoch_location_encoder(encoder_net, criterion, optimizer\
                                       , train_loader, training_epoch=True)
        tr_losses_history.append(tr_loss)    

        if epoch % logging_steps == 0:

            test_loss = run_one_epoch_location_encoder(encoder_net, criterion, optimizer\
                                       , test_loader, training_epoch=False)
            test_losses_history.append(test_loss)

        training_progress.set_description(f'Epoch {epoch +1} / {num_epochs}'+
              f'- Training loss {tr_loss.mean()**.5:.5f} - test loss {test_loss.mean()**.5:.5f}'  )

    trlh.append(tr_losses_history)
    tlh.append(test_losses_history)
    et = time.time()

    #save trained model:
    #0. create storage folder 
    models_path = data_path + "/models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Created data storage at:\n\t{models_path}")

    model_name = f"encoder_{num_epochs}.pt"
    trlh, tlh = np.vstack(trlh), np.vstack(tlh)

    import sys
    np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=500)
    training_info_df = pd.DataFrame({"classes_names":[classes_names]\
                  , "mpc":mpc\
                  , "non_empty_classes_names":[non_empty_classes_names]\
                  , "train_size": len(train_loader.dataset)
                  , "test_size": len(test_loader.dataset)
                  , "num_epochs":num_epochs\
                  , "pos_enc_dim" : pos_enc_dim
                  , "tsp":tsp\
                  , "enc_input_size":enc_input_size\
                  , "num_present_classes":num_present_classes\
                  , "xyz_centroid" :[norm_params[0]]\
                  , "xyz_max-min"  :[norm_params[1]]\
                  , "xyzh_centroid":[norm_params[2]]\
                  , "xyzh_max-min" :[norm_params[3]]\
                #   , "labels_centroid":[norm_params[4]]\
                #   , "labels_max-min" :[norm_params[5]]\
                  , "criterion":str(criterion)\
                  , "training_time_in_seconds": f"{int(et-st):,}"\
                  , "final_training_loss":tr_loss.mean()\
                  , "final_test_loss":test_loss.mean()\
                  , "training_losses_summary":[trlh[::max([1, num_epochs//10])]]\
                  , "test_losses_summary":[tlh[::max([1, num_epochs//20])]]\
                  , "training_losses_history":[trlh]\
                  , "test_losses_history":[tlh]}).T
    training_info_df.to_csv(f"{models_path}/training_info_{num_epochs}.csv")
    

    # Save
    #torch.save(encoder_net.state_dict(), f"{models_path}/{model_name}")
    #torch.save(encoder_net, f"{models_path}/{model_name}")
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(encoder_net.state_dict(), f"{models_path}/{model_name}")

    print(f"Model weights saved at:\n\t{models_path}/{model_name}")



    print(f"\nTraining for \n\t{num_epochs}     epochs took: \n\t{int(et-st):,}s, for an average of: \n\t{(et-st)/num_epochs:.1f}s per epoch.")

    if return_info:
        return encoder_net, trlh, tlh, visibility_dataset_df, training_info_df

    return encoder_net, trlh, tlh, visibility_dataset_df






############################
# Potential change in label scaling:

def process_locations_visibility_data_frame_with_labels(file_store, norm_params=None, min_percentage_per_class=.1, label_split=",", missing_labels=False, selected_label_indexes=None):
    '''
    Process locations.csv file from file_store. 
    Table expected columns:
        - x,y,z,xh,yh,zh,f_xyz
    Normalize coordinates and only consider non empty columns as labels.
    min_percentage_per_class - minimum percentage of a class to not be considered empty.
    Returns:
    processed_data_frame and indexes of non empty labels
    normalization_paramerers - (xyz_mean, xyz_dev, xyzh_mean, xyzh_dev)
    non_empty_classes        - array with true, false entries based on which f_xyz was kept.
    '''
    
    vis_df                        = pd.read_csv(file_store)
    
    if norm_params is None:
        vis_df_n, xyz_mean, xyz_dev   = normalize_visibility_dataframe(vis_df, ["x", "y", "z"])
        vis_df_n, xyzh_mean, xyzh_dev = normalize_visibility_dataframe(vis_df_n, ["xh", "yh", "zh"])
        norm_params = (xyz_mean, xyz_dev, xyzh_mean, xyzh_dev)
    else:
        xyz_mean, xyz_dev, xyzh_mean, xyzh_dev, label_mean, label_dev = norm_params
        vis_df_n, _, _  = normalize_visibility_dataframe(vis_df, ["x", "y", "z"], train_mean_dev=(xyz_mean, xyz_dev))
        vis_df_n, _, _  = normalize_visibility_dataframe(vis_df_n, ["xh", "yh", "zh"], train_mean_dev=(xyzh_mean, xyzh_dev))
    
    if missing_labels:
        return vis_df_n, norm_params, None
    
    if label_split is None:
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').replace("\n", "").split(" ") if d.isdigit()])
    if label_split == ",":
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').split(",") if d.isdigit()])

    #Normalize Labels by the sum of each row. Predictions will be adding up to 1
    max_row_value                      = sum(vis_df["f_xyz_raw"].iloc[0])

    # Filter labels either by index or by occurence
    if selected_label_indexes is None:
        #a. Filter down to only labels that appear:
        minimum_occurances                 = min_percentage_per_class * max_row_value * vis_df_n.shape[0]
        indvidual_class_occurences         = np.sum(np.vstack((vis_df_n["f_xyz_raw"].values)), axis=0)
        non_empty_classes                  = indvidual_class_occurences > minimum_occurances
    else:
        #b. Filter by selected indexes.
        non_empty_classes = np.in1d(np.arange(len(vis_df_n["f_xyz_raw"].iloc[0])), selected_label_indexes)

    #Keep only labels satisfying condition (occurences or selection)
    vis_df_n["f_xyz"]     = vis_df["f_xyz_raw"].apply(lambda d: \
                                             [p for (p, e) in zip(d, non_empty_classes) if e])
    #Normalize appearances:
    #vis_df_n["f_xyz"]     = vis_df["f_xyz"].apply(lambda d: [(2 * x) / max_row_value - 1 for x in d]) 
    #print(vis_df_n["f_xyz"])
    train_mean_dev = None if len(norm_params) == 4 else (label_mean, label_dev)
    vis_df_n, label_mean, label_dev = normalize_visibility_labels(vis_df_n, train_mean_dev=train_mean_dev)
    norm_params = norm_params + (label_mean, label_dev)
    
    #print(vis_df_n["f_xyz"])
    return vis_df_n, norm_params, non_empty_classes



def normalize_visibility_labels(vis_df, train_mean_dev=None):
    '''
    vis_df to be normalized on "f_xyz" column - independent feature-wise normalization
    Mean normalization:
        https://en.wikipedia.org/wiki/Feature_scaling#Methods
        
    train_mean_dev - (mean, std) where mean and std are for each feature
    '''
    label_matrix = np.vstack(vis_df["f_xyz"])
    #print(label_matrix)
    if train_mean_dev is None:
        label_mean   = label_matrix.mean(axis=0)
        label_max    = label_matrix.max(axis=0)
        label_min    = label_matrix.min(axis=0)
        label_dev    = (label_max - label_min)#.max()
        #print(label_mean, label_dev)
    else:
        label_mean, label_dev = train_mean_dev
    #return label_matrix, label_mean, label_dev
    vis_df["f_xyz_prenorm"] = vis_df["f_xyz"]
    #print(label_matrix, label_mean, label_dev)
    #print(label_matrix - label_mean, label_dev)
    #print((label_matrix - label_mean) / label_dev)
    vis_df["f_xyz"]       = ((label_matrix - label_mean) / label_dev).tolist()
    
    return vis_df, label_mean, label_dev


################################

def process_locations_visibility_data_frame_DEPRECARED(file_store, min_percentage_per_class=.1, label_split=None, missing_labels=False, selected_label_indexes=None):
    '''
    Process locations.csv file from file_store. 
    Table expected columns:
        - x,y,z,xh,yh,zh,f_xyz
    Normalize coordinates and only consider non empty columns as labels.
    min_percentage_per_class - minimum percentage of a class to not be considered empty.
    Returns:
    processed_data_frame and indexes of non empty labels
    '''
    
    visibility_dataset_df = pd.read_csv(file_store)#, index_col=0)
    
    #predicted_zooms = len(eval(visibility_dataset_df["f_xyz"].values[0])) - 1#.shape

    #Normalize xyz coordinates - substract mean and divide by max
    visibility_dataset_df[["xn", "yn", "zn"]] = visibility_dataset_df[["x", "y", "z"]] - np.mean(visibility_dataset_df.values[:,:3], axis=0)
    visibility_dataset_df[["xn", "yn", "zn"]] = visibility_dataset_df[["xn", "yn", "zn"]] / visibility_dataset_df[["xn", "yn", "zn"]].max().max()
    #Normalize xyz angles / view directions
    visibility_dataset_df[["xhn", "yhn", "zhn"]] = visibility_dataset_df[["xh", "yh", "zh"]]- np.mean(visibility_dataset_df.values[:,3:6], axis=0)
    #print(visibility_dataset_df[["xhn", "yhn", "zhn"]] .max())# Bug detected as angle normalization was performed with location mean [:,:3] instead of [:,3:6].
    visibility_dataset_df[["xhn", "yhn", "zhn"]] = visibility_dataset_df[["xhn", "yhn", "zhn"]] / visibility_dataset_df[["xhn", "yhn", "zhn"]].max().max()
    #print(visibility_dataset_df[["xhn", "yhn", "zhn"]] .max())
    if missing_labels:
        return visibility_dataset_df

    #Read list column:
    if label_split is None:
        visibility_dataset_df["f_xyz_raw"] = visibility_dataset_df["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').replace("\n", "").split(" ") if d.isdigit()])
    if label_split == ",":
        visibility_dataset_df["f_xyz_raw"] = visibility_dataset_df["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').split(",") if d.isdigit()])
    
    max_row_value                      = sum(visibility_dataset_df["f_xyz_raw"].iloc[0])

    # Filter labels either by index or by occurence
    if selected_label_indexes is None:
        #a. Filter down to only labels that appear:
        minimum_occurances                 = min_percentage_per_class * max_row_value * visibility_dataset_df.shape[0]
        indvidual_class_occurences         = np.sum(np.vstack((visibility_dataset_df["f_xyz_raw"].values)), axis=0)
        non_empty_classes                  = indvidual_class_occurences > minimum_occurances
    else:
        #b. Filter by selected indexes.
        non_empty_classes = np.in1d(np.arange(len(visibility_dataset_df["f_xyz_raw"].iloc[0])), selected_label_indexes)

    visibility_dataset_df["f_xyz"]     = visibility_dataset_df["f_xyz_raw"].apply(lambda d: \
                                             [p for (p, e) in zip(d, non_empty_classes) if e])
    #Normalize appearances:
    visibility_dataset_df["f_xyz"]     = visibility_dataset_df["f_xyz"].apply(lambda d: [(2 * x) / max_row_value - 1 for x in d]) 
    
    return visibility_dataset_df, non_empty_classes