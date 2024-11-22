"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
Architecture from: https://github.com/DveloperY0115/torch-NeRF/tree/main
"""

from tkinter import image_names
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np


from utils.scripts.architectures.torch_nerf_src import signal_encoder
###NeRFS - NerF on Surface - Nerf on paramteric surface


################################################################################
############## Custom Index formulas parsing to tensor operations ##############
################################################################################
import re
def expression_to_tensor_op(expression):
    '''
        tranforms id operations to torch tensor operation on vecotr with ids
        "1 / ( 1 + 4 )" -> tensor_operation {tensor -> tensor[1]/(tensor[1] + tensor[4])}
        Usage of returned operation: apply tensor_operation on encoder_outputs
        result = tensor_operation(encoder_outputs)
    '''
    # Replace each number in the expression with a tensor indexing operation
    expr_with_placeholders = re.sub(r'(\d+)', lambda x: f'encoder_outputs[{x.group(1)}]', expression)
    
    # Define a function that performs the operation when called
    def tensor_operation(encoder_outputs):
        # Safely evaluate the modified expression with encoder_outputs
        return eval(expr_with_placeholders, {"encoder_outputs": encoder_outputs})
    
    # Return the tensor operation as a callable function
    return tensor_operation

def category_to_index_expression(category_expression, category_names):
    """
    Transoform category expression string to logits index expression string.
    Example:
    "water / ( water + surface )" -> '1 / ( 1 + 4 )'
    """
    
    cat_exp_tokens = category_expression.split(" ")
    #Replace categories with indexes
    ind_exp_tokens = [str(category_names.index(s)) if s in category_names else s for s in cat_exp_tokens]
    
    index_expression = " ".join(ind_exp_tokens)
    
    return index_expression



################################################################################
##### NeRFS and parmateric surface classes:
################################################################################
# from utils.geometry_utils import surface_parametric
from utils.scripts.architectures.torch_nerf_src import signal_encoder
import torch
import torch.nn as nn

class NeRFS(nn.Module):
    """
    NeRFS - NeRF on a surface with parametric imputs.
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
        p: torch.Tensor=torch.empty(0),
        c: torch.Tensor=torch.empty(0),
        r: torch.Tensor=torch.empty(0),
        norm_params: tuple = ((0,0,0), (0,0,0)),
        pos_dim: int = 10,
        output_dim: int = 3,
        view_dir_dim: int = 3,
        feat_dim: int = 256,
        surface_type: str = "square",
        custom_formula_strings: list = []
    ):
        """
        Constructor of class 'NeRF'.

        p : point on the surface
        c : either (direction1, direction2) generating the plane or normal vector on the surface
        r : (l, L) width and breadth of the plane.

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

        #Hardcoded positional_encoder
        self.positional_encoder  = signal_encoder.positional_encoder.PositionalEncoder(pos_dim + view_dir_dim, 10, False)#, return_raw=return_raw)

        self.surface_type = surface_type
        self.norm_params  = norm_params
        self.p = p
        self.c = c
        self.r = r
        #self.surface_parametric = surface_parametric

        ##Custom perceptions tensor opearions:
        # e.g. "1 / ( 1 + 4 )" -> tensor_operation {tensor -> tensor[1]/(tensor[1] + tensor[4])}
        self.custom_formula_strings = custom_formula_strings
        self.custom_formula_tensors = []
        if len(self.custom_formula_strings) > 0:
            self.category_names = ['building', 'water', 'road', 'sidewalk', 'surface', 'tree', 'sky', "miscellaneous"]
            self.custom_formula_tensors = []
            for ce in self.custom_formula_strings:
                #Create the index operation string
                index_expression    = category_to_index_expression(ce, self.category_names)
                # Create the tensor operation
                tensor_operation = expression_to_tensor_op(index_expression)
                self.custom_formula_tensors.append(tensor_operation)


        print("Constructed tensor formulas:", self.custom_formula_tensors)

        try:
            self.surface = ParametricSurface(self.p,self.c,self.r,self.surface_type)
        except:
            print("Surface not initialized properly. Can only predict from raw position and direction.")
        
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        view_dir: torch.Tensor,
        on_surface: torch.Tensor=torch.empty(0),
        return_latent_features: bool = False
    ) -> torch.Tensor:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos (torch.Tensor): 

        Returns:
            sigma (torch.Tensor): Tensor of shape (N, ).
        """
        
        #raw_pos = surface_parametric(a, b, self.p, self.c, self.r, self.surface_type) 
        raw_pos = self.surface.parametrize(a,b)
        raw_view = view_dir
        #print(pos.dtype)
        
        #1. Normalize position and direction
        
        norm_pos = (raw_pos - self.norm_params[0]) / self.norm_params[1]
        pos      = norm_pos.unsqueeze(0)
        view_dir = (view_dir - self.norm_params[2]) / self.norm_params[3]
        view_dir = view_dir.unsqueeze(0)
        
        #print("Normalized inputs:", pos, view_dir)
        input_matrix = torch.vstack([pos[:, 0], pos[:, 1], pos[:, 2]\
                         , view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]]).T
        
        encoded_input = self.positional_encoder.encode((input_matrix))

        pos          = encoded_input[:,:encoded_input.shape[1]//2]
        view_dir     = encoded_input[:,encoded_input.shape[1]//2:]    
        #print(pos.dtype)
        #print("Encoded inputs:", pos, view_dir)


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
        latent_features = x
        rgb = self.sigmoid_actvn(self.fc_out(x))

        #Apply custom formulas if they were defined in __init__
        if len(self.custom_formula_tensors) > 0:
            # print("\nRaw semantics were: ", rgb.shape, list(zip(self.category_names, rgb[0])))
            #print(rgb) rgb[0] -- Assumption that the prediction is not on batches / batch size is 1.
            rgb = torch.vstack([cft(rgb[0]) for cft in self.custom_formula_tensors]).T#.flatten()
            # print("The predicted formulas are: ", self.custom_formula_strings)
        # print("The predictions are: ", rgb.shape, rgb)
            

        if return_latent_features:
            return (raw_pos, raw_view), latent_features, rgb

        #return rgb
        return raw_pos, raw_view, rgb
        #return sigma, rgb

    def predict_from_raw(
        self,
        position: torch.Tensor,
        view_dir: torch.Tensor,
        return_latent_features: bool = False
    ) -> torch.Tensor:
        """
            Same as forward but without paramterization.
            return_latent_features - returns also latent embedding x aside of rgb - same as get_latent_feature from NeRF
        """
        
        raw_pos = position
        raw_view = view_dir
        #print(pos.dtype)
        
        #1. Normalize position and direction
        
        norm_pos = (raw_pos - self.norm_params[0]) / self.norm_params[1]
        pos      = norm_pos.unsqueeze(0)
        view_dir = (view_dir - self.norm_params[2]) / self.norm_params[3]
        view_dir = view_dir.unsqueeze(0)
        
        #print("Normalized inputs:", pos, view_dir)
        input_matrix = torch.vstack([pos[:, 0], pos[:, 1], pos[:, 2]\
                         , view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]]).T
        
        encoded_input = self.positional_encoder.encode((input_matrix))

        pos          = encoded_input[:,:encoded_input.shape[1]//2]
        view_dir     = encoded_input[:,encoded_input.shape[1]//2:]    
        #print(pos.dtype)
        #print("Encoded inputs:", pos, view_dir)


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
        latent_features = x
        rgb = self.sigmoid_actvn(self.fc_out(x))

        #Apply custom formulas if they were defined in __init__
        if len(self.custom_formula_tensors) > 0:
            rgb = torch.vstack([cft(rgb) for cft in self.custom_formula_tensors])
        
        if return_latent_features:
            return (raw_pos, raw_view), latent_features, rgb
        #return rgb
        return raw_pos, raw_view, rgb
        #return sigma, rgb


class ParametricSurface:
    """
    Base class for Parametric Surfaces.
    optional_directions to initialize 
    """

    def __init__(self, p, c, r, surface_type):
        self.p            = p#torch.tensor(p)
        self.c            = c#torch.tensor(c)
        self.r            = r#torch.tensor(r)
        self.surface_type = surface_type
        # if self.surface_type == "semisphere":
        #     self.parametrize = self.parametrize_square
        if not isinstance(c, tuple): #plane defined by point on plane and point outside plane
            pc      = (c - p) + 1e-5
            pc_norm = torch.norm(pc)

            pc      = pc / pc_norm
            self.pc = pc
            orient  = torch.tensor([1, 1, (-pc[0] - pc[1]) / pc[2]])
            # orient  = torch.tensor([-1, 1, (pc[0] - pc[1]) / pc[2]])
            self.orient  = orient / torch.norm(orient)

            # Normalize pc to get the direction
            self.pc_normalized = pc / pc_norm

            # Find a vector perpendicular to pc
            self.v_perpendicular = torch.cross(self.pc_normalized, self.orient)

            # Normalize v_perpendicular
            self.v_perpendicular_normalized = self.v_perpendicular / torch.norm(self.v_perpendicular)
        else: #plane defined by center and two direction vectors.
            optional_directions  = c
            orient               = torch.tensor(optional_directions[1])
            self.orient          = orient / torch.norm(orient)
            self.v_perpendicular = torch.tensor(optional_directions[0])
            self.v_perpendicular_normalized = self.v_perpendicular / torch.norm(self.v_perpendicular)
            l, L = self.r
            self.p = self.p - 0.5 * l * self.v_perpendicular_normalized - 0.5 * L * self.orient# Translate center to the one corner of the plane
        #print(self.orient.round(), self.v_perpendicular_normalized.round())

        if self.surface_type == "square":
            self.parametrize = self.parametrize_square
        if self.surface_type == "sphere":
            self.parametrize = self.parametrize_sphere
            self.r           = pc_norm
            self.a_section   = torch.tensor(1)
            self.b_section   = torch.tensor(1)
        if self.surface_type == "semisphere":
            self.parametrize = self.parametrize_sphere
            self.r           = pc_norm
            self.a_section   = torch.tensor(r[0])
            self.b_section   = torch.tensor(r[1])
        
        
    def parametrize_sphere(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        theta = self.a_section * a * torch.pi     #between 0 and pi
        phi   = self.b_section * b * 2 * torch.pi #between 0 and 2*pi 

        x = self.r * torch.sin(theta) * torch.cos(phi)
        z = self.r * torch.sin(theta) * torch.sin(phi)
        y = self.r * torch.cos(theta)

        final_point = self.c + torch.stack([x, y, z])
        return final_point
        #final_point = c + torch.tensor([r * torch.sin(theta) * torch.cos(phi),0,0])
        #final_point = final_point + torch.tensor([0, r * torch.sin(theta) * torch.sin(phi),0])
        #final_point = final_point + torch.tensor([0, 0, r * torch.cos(theta)])
       

    def parametrize_square(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        #p,c,r,surface_type = self.p,self.c,self.r,self.surface_type
    
        #a = a.clamp(0,1); b = b.clamp(0,1)
        l, L = self.r
        a    = l * a#  * 2
        b    = L * b#  * 2 

        square_point = self.p + a * self.v_perpendicular_normalized + b * self.orient 
        final_point  = square_point
        return final_point

# ParametricSurface(p,c,r,surface_type).parametrize(a,b)



################################################################################
################################################################################

#Dataset needs to have # surface_parametric() - a,b,p,c,r,surface_type, lL:,
class EncoderNeRFSDataset(Dataset):
    #def __init__(self, parameters=None, surface=None, norm_params=None):
    def __init__(self, a, b, p, c, r, view_dir, targets, surface_type, norm_params=None):
        '''a,b paramters; p-point on surface;l c-center or point of reference; r- radius'''
        self.a = torch.autograd.Variable(a, requires_grad=True)   
        self.b = torch.autograd.Variable(b, requires_grad=True) 
        self.p = p#torch.tensor(p)#, requires_grad=True) 
        self.c = c#torch.tensor(c)#, requires_grad=True) 
        self.r = r#torch.tensor(r)#, requires_grad=True) 
        
        self.surface_type = surface_type
        self.norm_params  = norm_params
        self.view_dir     = view_dir
        self.targets      = (targets * 2) - 1 #same as: (targets - .5) * 2
        
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/2        
        self.a.retain_grad()
        self.b.retain_grad()
        #self.view_dir.retain_grad()
        #self.targets.retain_grad()
    
    def __getitem__(self,index):
        
        a, b, p, c, r, view_dir, targets, surface_type, norm_params = self.a, self.b, self.p, self.c, self.r, self.view_dir, self.targets, self.surface_type, self.norm_params
                
        sample = {"a":a, "b":b, "p":p, "c":c, "r":r, "view_dir":view_dir, "output":targets,  "surface_type":surface_type, "norm_params":norm_params}
        
        return sample

    def __len__(self):
        return len(self.a)