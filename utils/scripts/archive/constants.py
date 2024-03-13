# VOXEL_SIZE = .002 #Number of Meters (unnormalized scene) - Note that, each tile is approximately 400 × 400 square meters.
VOXEL_SIZE = 1 #Number of Meters (unnormalized scene) - Note that, each tile is approximately 400 × 400 square meters.
# VOXEL_SIZE = .002 * 2
#Learn-to-Score hyperparameters
Dx, Dy, Dz = 16, 16, 8
SCALES     = [0, 1, 2]


NUM_POINTS   = 10**6
NUM_POINTS   = 10
SAMPLE_SCENE = False
# SAMPLE_SCENE = True


STEP_SIZE     = 6
# STEP_SIZE     = 20
# STEP_SIZE     = 35

DATASET_SIZE = 2**14


##################################
########### PATHS: ###############
##################################

NP_FILES_PATH = "./np_files/"

#RT Utility / Visibility Training Scenes:
# SCENE_NAME = "birmingham_block_9"     #75 M POINTS #Field, highway buidlings
# SCENE_NAME = "cambridge_block_7"      #130 m points   # has churches and historical buidings
# SCENE_NAME = "cambridge_block_20"    #116 m points
# SCENE_NAME = "cambridge_block_33"     #27 mil points  # House Neigborhood 

# Test Scenes:
# SCENE_NAME = "birmingham_block_12"    #10 m points    # Long Stripe scene
SCENE_NAME = "birmingham_block_7"     #20 mil points  # Stadiums, halls, parking lots, roads 
# SCENE_NAME = "cambridge_block_3"        #102 m points   #Triangle shaped with small disjoint points island 
# SCENE_NAME = "cambridge_block_8"      #140 m points   #xCambridge King's College Chapel, churches similar to Westminster
# SCENE_NAME = "cambridge_block_14"      #125 m p oints   # Houses and Park


# SCENE_NAME = "birmingham_block_13"    #2  m points    # Parking Lot #Step Size 35
# SCENE_NAME = "cambridge_block_19" #85 M POINTS # Open field and houses


PLY_FILE_NAME = f"../datasets/SensatUrban_Dataset/ply/train/{SCENE_NAME}.ply"

# PLY_FILE_NAME = f"../datasets/Cloud_Gate_Open_Topography_PCD.ply"


MODEL_ARCHITECTURE_PATH = f'./{NP_FILES_PATH}/models/utility_model_RT_150.h5'
MODEL_WEIGHTS_PATH      = f"./{NP_FILES_PATH}/models/utility_models_RT"
