#Scripts inspired from `2_Directional_Semantic_Distribution_Model_Training.ipynb`
'''
    Directional model to predict information visible on a flat screen.
'''

from constants_direction import *
from directed_utility_inputs_4 import get_rotation_angles
from semantic_direction_7 import get_directional_nn_input, normalize_binary

from keras.models import load_model, model_from_json
from tensorflow import keras
from tqdm import tqdm
import numpy as np

def load_utility_model_from_path(model_path, weights_path, version=3):
    '''
        version 2 - RT utility (1 prediction);
        version 3 - semantic utility prediction without regularizing sum. (5 predictions);
        version 4 - not working for the moment - five predictions plus regularizing sum;
        version 5 - to be implemented. 5 v2 models to predict semantic utility per class.
    '''
    
    pre_compile = False if version == 3 else True
    model_loaded = load_model('models/v3/log_utility_model_RT_Directed_Semantic_50_3.h5', compile=pre_compile)
    model_loaded.load_weights('models/v3/log_utility_model_RT_Directed_Semantic_50_weights_3.h5')
    
    if version == 3:
        losses = {
        "predictions_total": "MeanSquaredError",
        "predictions_ground": "MeanSquaredError",
        "predictions_immobile": "MeanSquaredError",
        "predictions_mobile": "MeanSquaredError",
        "predictions_nature": "MeanSquaredError",
        #     "zero_sum": "MeanSquaredError"
        }    
    
        model_loaded.compile(
            optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            # loss=keras.losses.MeanSquaredError(),
            loss=losses,
            # List of metrics to monitor
            metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()],
        )
    return model_loaded


def get_model_predicted_utilities(potential_locations, model, voxel_grid\
                                  , num_rotations=NUM_ROTATIONS, depth=DEPTHS[-1]):
    utility_values     = np.zeros((num_rotations * potential_locations.shape[0], 5))
    ploc_inputs        = np.zeros(shape=(num_rotations * potential_locations.shape[0]\
                                        , depth//2, depth//4, depth//4, 3))
    directions         = []

    for i, p_loc in tqdm(enumerate(potential_locations), total=potential_locations.shape[0]):
        for nr in range(num_rotations):
            custom_direction   = get_rotation_angles(random_direction=True)
            directions.append(custom_direction)
            
            #input from location and direction:
            directed_input      = get_directional_nn_input(location=p_loc, direction=custom_direction\
                                                         , voxel_grid=voxel_grid, depth=depth)
            normalized_input    = normalize_binary(directed_input)
            ploc_input          = np.moveaxis(normalized_input, 0, -1)[np.newaxis,:,:,:,:]
            ploc_inputs[nr + i * num_rotations] = ploc_input
            
            #
            predicted_utility          = model.predict(ploc_input, verbose=0)
            utility_values[nr + i * num_rotations]     = np.concatenate(predicted_utility).reshape(-1)

    return utility_values, np.repeat(potential_locations, num_rotations, axis=0), directions, ploc_inputs