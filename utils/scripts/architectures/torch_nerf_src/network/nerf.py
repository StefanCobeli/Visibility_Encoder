"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
Architecture from: https://github.com/DveloperY0115/torch-NeRF/tree/main
"""

from tkinter import image_names
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset



from utils.scripts.architectures.torch_nerf_src import signal_encoder


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        output_dim: int = 3,
        view_dir_dim: int = 3,
        feat_dim: int = 256,
    ):
        """
        Constructor of class 'NeRF'.

        Args:
            pos_dim (int): Dimensionality of coordinate vectors of sample points.
            view_dir_dim (int): Dimensionality of view direction vectors.
            feat_dim (int): Dimensionality of feature vector within forward propagation.
                Set to 256 by default following the paper.
        """
        super().__init__()

        # rgb_dim = 3
        rgb_dim = output_dim
        self.output_dim = output_dim
        density_dim = 1

        self._pos_dim = pos_dim
        self._view_dir_dim = view_dir_dim
        self._feat_dim = feat_dim

        # fully-connected layers
        self.fc_in = nn.Linear(self._pos_dim, self._feat_dim)
        self.fc_1 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_2 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_3 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_4 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_5 = nn.Linear(self._feat_dim + self._pos_dim, self._feat_dim)
        self.fc_6 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_7 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_8 = nn.Linear(self._feat_dim, self._feat_dim + density_dim)
        self.fc_9 = nn.Linear(self._feat_dim + self._view_dir_dim, self._feat_dim // 2)
        self.fc_out = nn.Linear(self._feat_dim // 2, rgb_dim)

        # activation layer
        self.relu_actvn = nn.ReLU()
        # self.sigmoid_actvn = nn.Sigmoid()
        self.sigmoid_actvn = nn.Tanh()

    def get_latent_feature(self,
        pos: torch.Tensor,
        view_dir: torch.Tensor):
        '''same as forward but returns also latent embedding x aside of rgb'''
        if pos.shape[0] != view_dir.shape[0]:
            raise ValueError(
                f"The number of samples must match. Got {pos.shape[0]} and {view_dir.shape[0]}."
            )
        if pos.shape[-1] != self._pos_dim:
            raise ValueError(f"Expected {self._pos_dim}-D position vector. Got {pos.shape[-1]}.")
        # if view_dir.shape[-1] != self._view_dir_dim:
        #     raise ValueError(
        #         f"Expected {self._view_dir_dim}-D view direction vector. Got {view_dir.shape[-1]}."
        #     )

        x = self.relu_actvn(self.fc_in(pos))
        x = self.relu_actvn(self.fc_1(x))
        x = self.relu_actvn(self.fc_2(x))
        x = self.relu_actvn(self.fc_3(x))
        x = self.relu_actvn(self.fc_4(x))

        x = torch.cat([pos, x], dim=-1)

        x = self.relu_actvn(self.fc_5(x))
        x = self.relu_actvn(self.fc_6(x))
        x = self.relu_actvn(self.fc_7(x))
        x = self.fc_8(x)

        #sigma = self.relu_actvn(x[:, 0])
        x = torch.cat([x[:, 1:], view_dir], dim=-1)

        x = self.relu_actvn(self.fc_9(x))
        rgb = self.sigmoid_actvn(self.fc_out(x))
        return rgb, x 
    '''
    Abandoned:
    def get_prediction_from_raw_xyzca(self,
        pos: torch.Tensor,
        view_dir: torch.Tensor):

        input_matrix   = np.vstack([vis_df[n[0]], vis_df[n[1]], vis_df[n[2]]\
                                    , vis_df[n[3]], vis_df[n[4]], vis_df[n[5]]]).T.astype(np.float32)

        encoded_input = self.positional_encoder.encode(torch.tensor(input_matrix))
        input_pos     = encoded_input[:,:encoded_input.shape[1]//2]
        input_dir     = encoded_input[:,encoded_input.shape[1]//2:]

        signal_encoder.positional_encoder.PositionalEncoder(xyz_dim, pos_enc_dim, False)
    '''

    def forward(
        self,
        pos: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos (torch.Tensor): Tensor of shape (N, self.pos_dim).
                Coordinates of sample points along rays.
            view_dir (torch.Tensor): Tensor of shape (N, self.dir_dim).
                View direction vectors.

        Returns:
            sigma (torch.Tensor): Tensor of shape (N, ).
                Density at the given sample points.
            rgb (torch.Tensor): Tensor of shape (N, 3).
                Radiance at the given sample points.
        """
        # check input tensors
        # if (pos.ndim != 2) or (view_dir.ndim != 2):
        #     raise ValueError(f"Expected 2D tensors. Got {pos.ndim}, {view_dir.ndim}-D tensors.")
        if pos.shape[0] != view_dir.shape[0]:
            raise ValueError(
                f"The number of samples must match. Got {pos.shape[0]} and {view_dir.shape[0]}."
            )
        if pos.shape[-1] != self._pos_dim:
            raise ValueError(f"Expected {self._pos_dim}-D position vector. Got {pos.shape[-1]}.")
        # if view_dir.shape[-1] != self._view_dir_dim:
        #     raise ValueError(
        #         f"Expected {self._view_dir_dim}-D view direction vector. Got {view_dir.shape[-1]}."
        #     )

        x = self.relu_actvn(self.fc_in(pos))
        x = self.relu_actvn(self.fc_1(x))
        x = self.relu_actvn(self.fc_2(x))
        x = self.relu_actvn(self.fc_3(x))
        x = self.relu_actvn(self.fc_4(x))

        x = torch.cat([pos, x], dim=-1)

        x = self.relu_actvn(self.fc_5(x))
        x = self.relu_actvn(self.fc_6(x))
        x = self.relu_actvn(self.fc_7(x))
        x = self.fc_8(x)

        #sigma = self.relu_actvn(x[:, 0])
        x = torch.cat([x[:, 1:], view_dir], dim=-1)

        x = self.relu_actvn(self.fc_9(x))
        rgb = self.sigmoid_actvn(self.fc_out(x))

        return rgb
        #return sigma, rgb

    @property
    def pos_dim(self) -> int:
        """Returns the acceptable dimensionality of coordinate vectors."""
        return self._pos_dim

    @property
    def view_dir_dim(self) -> int:
        """Returns the acceptable dimensionality of view direction vectors."""
        return self._view_dir_dim

    @property
    def feat_dim(self) -> int:
        """Returns the dimensionality of internal feature vectors."""
        return self._feat_dim



################################################################################
################ 1. Encoder only with Location
################################################################################
class EncoderNeRFDataset(Dataset):
    def __init__(self,vis_df=None, label_column_name="f_xyz_rounded", features_column_names=["x", "y", "z", "xh", "yh", "zh"], pos_enc_dim=10, missing_labels=False):

        #pos_enc_dim                = 10   # 4 or 10 #See NeRF paper section 5.1 Positional encoding, page 8 - L = 4 or L=10 for γ(d).
        xyz_dim                    = 6 # 6 - x,y,z,xh,yh,zh
        # Setup positional encoder:
        self.positional_encoder        = signal_encoder.positional_encoder.PositionalEncoder(xyz_dim, pos_enc_dim, False)
        self.features_column_names     = features_column_names
        self.missing_labels            = missing_labels

        if vis_df is not None:
            self.input_pos, self.input_dir = self.encode_position(vis_df, n=features_column_names)

            self.image_name = vis_df["image_name"].tolist()
            
            if not(self.missing_labels ):
                output_matrix  = np.vstack([eval(vd) if type(vd) == str else vd for vd in vis_df[label_column_name]]).astype(np.float32)
            
                self.output    = torch.tensor(output_matrix)

    def encode_position(self, vis_df, n=None):

        if n is None:
            n= self.features_column_names
        input_matrix   = np.vstack([vis_df[n[0]], vis_df[n[1]], vis_df[n[2]]\
                                    , vis_df[n[3]], vis_df[n[4]], vis_df[n[5]]]).T.astype(np.float32)
        encoded_input = self.positional_encoder.encode(torch.tensor(input_matrix))
        input_pos     = encoded_input[:,:encoded_input.shape[1]//2]
        input_dir     = encoded_input[:,encoded_input.shape[1]//2:]

        return input_pos, input_dir

    
    def encode_np_locations(self, xyz, xyzh):

        input_matrix   = np.hstack([xyz, xyzh]).T.astype(np.float32)
        encoded_input = self.positional_encoder.encode(torch.tensor(input_matrix))
        input_pos     = encoded_input[:,:encoded_input.shape[1]//2]
        input_dir     = encoded_input[:,encoded_input.shape[1]//2:]

        return input_pos, input_dir

    def __getitem__(self,index):
        input_pos, input_dir  = self.input_pos[index], self.input_dir[index]
        if self.missing_labels:
            sample = {'input_pos': input_pos, 'input_dir': input_dir, "image_name": self.image_name[index]}
        else:
            output = self.output[index]
            sample = {'input_pos': input_pos, 'input_dir': input_dir, "output" : output, "image_name": self.image_name[index]}
        return sample

    def __len__(self):
        return len(self.input_pos)



################################################################################
################ 2. DEPRECATED Encoder with Location AND Visual ###########################
################################################################################
class EncoderNeRFDatasetVisual(Dataset):
    """!!!!!!!!!!!
    !!!!DEPRECATED
    !!!!!!!!!!!!!!
    """
    def __init__(self,vis_df, image_column_name="f_xyz_rounded", label_column_name="f_xyz_rounded", features_column_names=["x", "y", "z", "xh", "yh", "zh"]):
        pos_enc_dim                    = 4 #See nerf 5.1 Positional encoding, page 8 - L = 4 for γ(d).
        xyz_dim                        = 6 # 6 - x,y,z,xh,yh,zh
        self.positional_encoder        = signal_encoder.positional_encoder.PositionalEncoder(xyz_dim, pos_enc_dim, False)
        self.features_column_names     = features_column_names
            
        self.input_pos, self.input_dir = self.encode_position(vis_df, n=features_column_names)

        output_matrix  = np.vstack([eval(vd) if type(vd) == str else vd for vd in vis_df[label_column_name]]).astype(np.float32)
        self.output    = torch.tensor(output_matrix)

    def encode_position(self, vis_df, n=None):
        """n is a list with the feature names in vis_df"""
        if n is None:
            n= self.features_column_names
        input_matrix   = np.vstack([vis_df[n[0]], vis_df[n[1]], vis_df[n[2]]\
                                    , vis_df[n[3]], vis_df[n[4]], vis_df[n[5]]]).T.astype(np.float32)
        encoded_input = self.positional_encoder.encode(torch.tensor(input_matrix))
        input_pos     = encoded_input[:,:encoded_input.shape[1]//2]
        input_dir     = encoded_input[:,encoded_input.shape[1]//2:]

        return input_pos, input_dir

    def encode_np_locations(self, xyz, xyzh):

        input_matrix   = np.hstack([xyz, xyzh]).T.astype(np.float32)
        encoded_input = self.positional_encoder.encode(torch.tensor(input_matrix))
        input_pos     = encoded_input[:,:encoded_input.shape[1]//2]
        input_dir     = encoded_input[:,encoded_input.shape[1]//2:]

        return input_pos, input_dir

    def __getitem__(self,index):
        input_pos, input_dir, output = self.input_pos[index], self.input_dir[index], self.output[index]
        sample = {'input_pos': input_pos, 'input_dir': input_dir, "output":output}
        return sample

    def __len__(self):
        return len(self.input_pos)

#################################################################################################
################0. Version 1 of Location Encoder with min max ##################################
#################################################################################################

#Inspired from:
# https://github.com/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/MLP_1class_BinaryCrossEntropyLoss.ipynb
from torch import nn
from torch.utils.data import Dataset
import numpy as np

from sklearn.preprocessing import MinMaxScaler
class EncoderDataset(Dataset):
    def __init__(self,vis_df, scaler=None):
        input_matrix   = np.vstack([vis_df["x"], vis_df["y"], vis_df["z"]\
                                    , vis_df["xh"], vis_df["yh"], vis_df["zh"]]).T.astype(np.float32)
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            
        input_matrix = self.scaler.fit_transform(input_matrix)
        self.input     = torch.tensor(input_matrix)
        output_matrix  = np.vstack([eval(vd) for vd in vis_df["f_xyz_rounded"]]).astype(np.float32)
        self.output    = torch.tensor(output_matrix)

    def __getitem__(self,index):
        return self.input[index], self.output[index]
        #return self.x[index], self.y[index], self.z[index], self.xh[index],self.yh[index], self.zh[index]

    def __len__(self):
        return len(self.input)



#Deprecated - Repaced by NeRF
class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel,self).__init__()
        self.fc1 =torch.nn.Linear(6,2048)
        self.fc2 =torch.nn.Linear(2048,1024)
        self.fc3 =torch.nn.Linear(1024,512)
        self.fc4 =torch.nn.Linear(512,256)
        self.fc5 =torch.nn.Linear(256,128)
        self.fc6 =torch.nn.Linear(128,3)
        self.sigmoid=torch.nn.Sigmoid()
        
        self.relu = nn.ReLU()

    def forward(self,x):
        #print(x.dtype)
        out =self.fc1(x)
        out =self.relu(out)
        out =self.fc2(out)
        out =self.relu(out)
        out =self.fc3(out)
        out =self.relu(out)
        out =self.fc4(out)
        out =self.relu(out)
        out =self.fc5(out)
        out =self.relu(out)
        out =self.fc6(out)
        out= self.sigmoid(out)
        return out
     