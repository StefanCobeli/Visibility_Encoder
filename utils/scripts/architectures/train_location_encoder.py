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
    #print(current_losses.shape)
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
        return mean_epoch_loss, np.concatenate(all_losses), np.concatenate(predictions)#,np.array(predictions)
    #print(len(data_loader.dataset), return_predictions, predictions)
    return mean_epoch_loss#, current_losses


# class ScaledWeightedMSELoss(torch.nn.Module):
#     def __init__(self, scale_factors, weights):
#         super().__init__()
#         self.scale_factors = scale_factors
#         self.weights = weights

#     def forward(self, y_pred, y_true):
#         return torch.mean(self.weights * ((y_pred - y_true) / self.scale_factors) ** 2)
'''Causes too extreme loss values: e.g. tensor([    0.0225,  1227.7540], dtype=torch.float64)'''
class ScaledWeightedMSELoss(torch.nn.Module):
    def __init__(self, scale_factors, weights, epsilon=1e-6, reduction='none'):
        super().__init__()
        self.scale_factors = scale_factors + epsilon  # Prevent div by zero
        self.weights = torch.clamp(weights, min=1e-3, max=10)  # Avoid extreme values
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = y_pred * .5 + .5
        y_true = y_true * .5 + .5

        loss = self.weights * ((y_pred - y_true) / self.scale_factors) ** 2
        # loss = self.weights * ((y_pred - y_true)) ** 2
        loss_not_nan = loss[~torch.isnan(loss)] # Ignore NaNs
        if self.reduction == "mean":
            return torch.mean(loss_not_nan) 
        if self.reduction == 'none':
            return loss_not_nan 
        return loss_not_nan # in any other case of reduction
        
    
def compute_scale_factors_and_weights(Y_train):
    Y_train = Y_train * .5 + .5
    scale_factors = torch.std(Y_train, dim=0) + 1e-6
    weights = 1.0 / (torch.mean(Y_train, dim=0) + 1e-6)
    print("\npercentage Scaling factor:", scale_factors)
    print("percentage Class weights:", weights, "\n")
    return scale_factors, weights


class BalancedMSELoss(torch.nn.Module):
    def __init__(self, targets, epsilon=1e-6, reduction='none'):
        """
        targets: Tensor of shape (N, D) representing the true target values
        epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        # Compute per-dimension inverse frequency-based weights
        target_var = torch.var(targets, dim=0, unbiased=True)  # Variance per dimension
        self.weights = 1.0 / (target_var + epsilon)  # Inverse variance weighting
        self.weights /= self.weights.sum()  # Normalize weights
        print("BalancedMSELoss\n\tWeights:", self.weights)
        self.reduction = reduction

    def forward(self, predictions, targets):
        loss = self.weights * (predictions - targets) ** 2
        if self.reduction=="none":
            return loss        
        return loss.mean()#if any other reduction is passed aside of none.


def get_location_visibility_encoder(enc_input_size, num_present_classes, feat_dim=256, output=None):
    '''
    Setup training model: 
        enc_input_size: stands for both the size of the location and of the direction.
        num_present_classes: size of the output layer.
        output is the entire y_train. If none is provided, then loss will be simpy MSELoss
    returns: encoder_net, criterion, optimizer, scheduler
    '''
    #Hyperparameters:
    lr_start            = 1e-5 
    
    #Model and loss declaration
    encoder_net = network.nerf.NeRF(pos_dim=enc_input_size, output_dim=num_present_classes, view_dir_dim=enc_input_size, feat_dim=feat_dim) 
    #! NeRFS is only used for testing not for training: The surface projection does not need to be trained.
    # encoder_net = network.nerfs.NeRFS(pos_dim=enc_input_size, output_dim=num_present_classes, view_dir_dim=enc_input_size, feat_dim=feat_dim) 
    
    if output is None: #Use simple MSE Loss
        criterion   = torch.nn.MSELoss(reduction='none')
    else:
        criterion = BalancedMSELoss(output, reduction='none')
    # else: # ScaledWeightedMSELoss produces extreme loss values: e.g. tensor([    0.0225,  1227.7540], dtype=torch.float64)
    #     scale_factors, weights = compute_scale_factors_and_weights(output)
    #     criterion = ScaledWeightedMSELoss(scale_factors, weights, reduction='none')
        
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

def process_locations_visibility_data_frame(file_store, norm_params=None, min_percentage_per_class=None, label_split=",", missing_labels=False, selected_label_indexes=None):#[0,1,5,6]):
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
    if ".json" in file_store:
        vis_df = pd.read_json(file_store)
        if label_split is not None:
            # label_name  = label_split
            label_name  = "f_xyz"
            label_split = "json"
    else:
        vis_df                                  = pd.read_csv(file_store)
    # vis_df[['x','y',"z", 'xh', 'yh', 'zh']] = vis_df[['x','y',"z", 'xh', 'yh', 'zh']].round(3)
    
    # if "f_xyz_raw" in vis_df:
    #     vis_df 

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
    
    max_row_value = None
    if label_split is None:
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').replace("\n", "").split(" ") if d.isdigit()])
    if label_split == ",":
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(lambda d: [eval(d) for d in d.strip('[]').split(",") if d.isdigit()])
    if label_split == "json":
        # vis_df_n["f_xyz"] = vis_df_n[label_name]
        vis_df_n["f_xyz_raw"] = vis_df_n["f_xyz"].apply(eval)
        #max_row_value     = vis_df_n[label_name].max() #works, assuming row with [0,..,0,max,0,..0]
        max_row_value     = sum(vis_df["f_xyz_raw"].iloc[0])

    if max_row_value is None:
        #Normalize Labels by the sum of each row. Predictions will be adding up to 1
        max_row_value                      = sum(vis_df["f_xyz_raw"].iloc[0])
    print(f"Normalizing each label value by: {max_row_value:,}")

    # Filter labels either by index or by occurence
    if selected_label_indexes is None:
        #a. Filter down to only labels that appear:
        if min_percentage_per_class is None: #all indexes are considered selected indexes
            # print(eval(vis_df_n["f_xyz"]))
            #vis_df_n["f_xyz"] = vis_df_n["f_xyz"].lambda(f: eval(f)) #evaluate the list of strings
            # print(vis_df_n["f_xyz"].values[0])
            # print(eval(vis_df_n["f_xyz"].values[0]))
            selected_label_indexes = np.arange(len(eval(vis_df_n["f_xyz"].iloc[0])))#evaluate the list of strings
            non_empty_classes      = np.ones_like(selected_label_indexes).astype(bool)
            print("Selected label indexes:", selected_label_indexes, non_empty_classes)
        else:
            minimum_occurances                 = min_percentage_per_class * max_row_value * vis_df_n.shape[0]
            indvidual_class_occurences         = np.sum(np.vstack((vis_df_n["f_xyz_raw"].values)), axis=0)
            non_empty_classes                  = indvidual_class_occurences > minimum_occurances
            print("Selected label indexes:", selected_label_indexes, non_empty_classes)
    else:
        #b. Filter by selected indexes.
        # print(selected_label_indexes)
        # print(np.arange(len(vis_df_n["f_xyz_raw"].iloc[0])))
        # print(np.in1d(np.arange(len(vis_df_n["f_xyz_raw"].iloc[0])), selected_label_indexes))
        non_empty_classes = np.in1d(np.arange(len(vis_df_n["f_xyz_raw"].iloc[0])), selected_label_indexes)
        # print(non_empty_classes)
    #Keep only labels satisfying condition (occurences or selection)
    vis_df_n["f_xyz"]     = vis_df["f_xyz_raw"].apply(lambda d: \
                                             [p for (p, e) in zip(d, non_empty_classes) if e])
    #Normalize appearances and strech them between -1, +1:
    vis_df_n["f_xyz"]     = vis_df["f_xyz"].apply(lambda d: [(2 * x) / max_row_value - 1 for x in d]) 
    
    # print(np.vstack(vis_df_n['f_xyz'].values).max(axis=1).shape, np.vstack(vis_df_n['f_xyz'].values).shape)
    print(f"\nNormalized f_xyz between -1 ({np.vstack(vis_df_n['f_xyz'].values).min(axis=0)}) "\
    f"and 1 ({np.vstack(vis_df_n['f_xyz'].values).max(axis=0)}), considering the maximum label row value of {max_row_value}")
    #If there are additional labels not to be normalized they can be added here.
    #labels to be steored in "f_xyz_2" or "f_xyz_3" and so on. new labels presumed to be normalized already.
    if "f_xyz_2" in vis_df.columns:
        vis_df_n["f_xyz"] = np.hstack([np.vstack(vis_df_n["f_xyz"].values), np.vstack(vis_df["f_xyz_2"].values)]).tolist()#[:2]#vis_depth_df.apply(lambda d: np.hstack([d["f_xyz"], d["f_xyz_2"]]), axis=1)
    

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


def train_model_on_data(data_path, num_epochs=200, tsp=1, mpc = None, separate_test=False, selected_label_indexes=None, file_name="locations", return_info=False, model_name=None, label_name=None):
    '''
    mpc: min_percentage_per_class - 0-1 how often a class should appear in the set to be considered - default .05.
    tsp: training set percentage  - out of the 80% of the data on how many samples should the model be trained - default 1.
    selected_label_indexes - [0, 1, ...] to ignore mpc

    returns: encoder_net, trlh, tlh, visibility_dataset_df training_info_df (training_info_df only if return_info)
    '''

    import time

    st = time.time()

    NY_MESH = True
    # NY_MESH = False

    if ".json" in data_path:
        file_store       = data_path #+ f"/{file_name}"
        data_path = "/".join(data_path.split("/")[:-1])
        class_index_file = data_path + "/classes_index.csv"
        models_path      = data_path + "/models"
    else:
        file_store      = data_path + f"/{file_name}.csv"
        class_index_file = f"{data_path}/classes_index.csv"
        models_path      = data_path + "/models"

    if separate_test:
        file_store_train = data_path + "/locations_train.csv"
        file_store_test  = data_path + "/locations_test.csv"
    
    if NY_MESH:

        try:
            ins_dict = {c["color"] : c["class"] for c in  \
                    pd.read_csv(class_index_file\
                                , index_col=False).to_dict(orient="records")
                    }
        except:
            print("Missing classes_index.csv file!!!")
            storage_folder = "/".join(data_path.split("/")[:-1])
            ins_dict = {c["color"] : c["class"] for c in  \
                    pd.read_csv(storage_folder + "/classes_index.csv"\
                                , index_col=False).to_dict(orient="records")
                    }    
            models_path      = storage_folder + "/models"
            file_store       = data_path 
        label_split=","
        classes_names = list(ins_dict.values())
    else:
        from utils.scripts.helper_ply_SensatUrban_0 import ins_dict
        label_split=None
        file_store = "./data/cambridge_block_8_fragment_semantic_locs-2500_dirs-5_visual/locations.csv"
        classes_names = list(ins_dict.values())+["empty"]

    if label_name is not None: #If there is a label name other than f_xyz.
        label_split = label_name 
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
    if mpc is not None:
        print(f"Found {nec.sum()} classes appearing more than {mpc*100:.1f}%: {non_empty_classes_names}")
    else:
        print(f"Train model based on selected label indexes: {non_empty_classes_names}")


    trlh = []
    tlh  = []
    tralh = []# all losses training history
    talh = [] # all losses testing history

    #2. setup `torch` dataset and loaders
    pos_enc_dim = 10
    # batch_size=16
    batch_size=8
    # batch_size=32
    # batch_size=1024
    train_loader, test_loader = get_location_visibility_loaders(processed_vis_loc_df=visibility_dataset_df\
    , train_set_percentage=tsp, test_size=0.2, batch_size=batch_size, pos_enc_dim=pos_enc_dim, seed=1)

    #3. Setup NeRF-like location encoder:
    enc_input_size, num_present_classes          = train_loader.dataset.input_dir.shape[1], train_loader.dataset.output.shape[1]
    # print(train_loader.dataset.output) #tensor with data_points x labels // can be seen as Y_train in a classical ML approach

    #print("Num present classes", num_present_classes)
    custom_loss = None
    if custom_loss:
        encoder_net, criterion, optimizer, scheduler =  get_location_visibility_encoder(enc_input_size, num_present_classes, output=train_loader.dataset.output)
    else:
        encoder_net, criterion, optimizer, scheduler =  get_location_visibility_encoder(enc_input_size, num_present_classes, output=None)
    #4. Training loop
    print(f"Training data percentace {100 * tsp:.2f}% - {int(tsp * len(visibility_dataset_df)*.8):,} samples, for {num_epochs} epochs:")
    training_progress   = tqdm(range(num_epochs))
    logging_steps       = 2#5
    tr_losses_history   = []
    test_losses_history = []
    all_test_losses = []
    all_tr_losses   = []
    for epoch in training_progress:

        tr_loss = run_one_epoch_location_encoder(encoder_net, criterion, optimizer\
                                       , train_loader, training_epoch=True, return_predictions=True)
        all_tr_losses.append(tr_loss[1])   
        tr_losses_history.append(tr_loss[0])  

        if epoch % logging_steps == 0:

            test_loss = run_one_epoch_location_encoder(encoder_net, criterion, optimizer\
                                       , test_loader, training_epoch=False, return_predictions=True)
            all_test_losses.append(test_loss[1])   
            test_losses_history.append(test_loss[0])  

            # reportable_metrics = 
            debug_training = None
            if debug_training:
                torch.set_printoptions(sci_mode=False)
                print("predictions shape:", test_loss[2].shape\
                , "prediction example:", .5 * test_loss[2][0] + .5\
                , "\nground truth shape:", test_loader.dataset.output.shape\
                , "ground truth example:", .5 * test_loader.dataset.output[0] + .5)
                print(
                "\nLoss example:"\
                ,  torch.nn.MSELoss(reduction='none')(\
                    torch.tensor(.5 * test_loss[2][0] + .5)\
                    , .5 * test_loader.dataset.output[0] + .5)\
                , "Total MSE Loss:"\
                ,   torch.nn.MSELoss(reduction='none')(\
                    .5 * torch.tensor(test_loss[2]) + .5\
                    , .5 * test_loader.dataset.output + .5).mean(axis=0)\
                ,"\nTotal BMSE Optimization Loss:"\
                , dict(zip(non_empty_classes_names, criterion(torch.tensor(test_loss[2]), test_loader.dataset.output).mean(axis=0)))
                )

        training_progress.set_description(f'Epoch {epoch +1} / {num_epochs}'+
              f'- Training loss {tr_loss[0].mean()**.5:.5f} - test loss {test_loss[0].mean()**.5:.5f}'  )

    trlh.append(tr_losses_history)
    tlh.append(test_losses_history)
    tralh.append(all_tr_losses)
    talh.append(all_test_losses)

    et = time.time()

    #save trained model:
    #0. create storage folder 
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Created data storage at:\n\t{models_path}")
    if model_name is None:
        model_name = f"encoder_{num_epochs}.pt"
        
    trlh, tlh = np.vstack(trlh), np.vstack(tlh)

    # print(norm_params[0],norm_params[1],norm_params[2],norm_params[3])
    print({"classes_names":[n.strip(" ") for n in classes_names]})

    import sys
    np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=500)
    training_info_df = pd.DataFrame({"classes_names":[[n.strip(" ") for n in classes_names]]\
                  , "mpc":mpc\
                  , "non_empty_classes_names":[[n.strip(" ") for n in non_empty_classes_names]]\
                  , "train_size": len(train_loader.dataset)
                  , "test_size": len(test_loader.dataset)
                  , "num_epochs":num_epochs\
                  , "pos_enc_dim" : pos_enc_dim
                  , "tsp":tsp\
                  , "batch_size": batch_size\
                  , "label_name": "f_xyz" if label_name is None else label_name\
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
                  , "final_training_loss":tr_loss[0].mean()\
                  , "final_test_loss":test_loss[0].mean()\
                  , "training_losses_summary":[trlh[::max([1, num_epochs//10])]]\
                  , "test_losses_summary":[tlh[::max([1, num_epochs//20])]]\
                  , "training_losses_history":[trlh]\
                  , "test_losses_history":[tlh]\
                #   , "all_training_losses_history" :tralh\
                #   , "all_test_losses_history" :talh\
                }).T
    training_info_df.to_csv(f"{models_path}/training_info_{num_epochs}.csv")
    training_info_df.to_json(f"{models_path}/training_info_{num_epochs}.json", indent=4)

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



###


############################

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

