import sys
sys.path.append("./scripts/")
from full_pipeline_360_scripts_6    import *
from constants                      import *

from grid_processing_scripts_1      import *
from scene_loading_2                import *
from ray_casting_3                  import *
from selection_algorithm_5          import *#check_clear_neighborhood

from directed_utility_inputs_4      import *

# import jax.numpy as np

from helper_ply_SensatUrban_0 import write_ply

# (400, 480 ), (1090, 1170)
def cut_large_ply(bx, by, PLY_FILE_NAME):
    """
        Cut ply file by x y constraints in rectangular shape.
        e.g. bx, by = (400, 480), (1090, 1170)
    """
    fl = DataProcessing().read_ply_data(PLY_FILE_NAME)
    selected_ply_path = f'{PLY_FILE_NAME.split("/")[-1][:-4]}_fragment.ply'
    selected_ids = np.where((fl[0][:,0] >= bx[0]) & (fl[0][:,0] <=bx[1]) \
                            & (fl[0][:,1] >= by[0]) & (fl[0][:,1] <=by[1]))# (400, 480 ), (1090, 1170)
    fl_selected = fl[0][selected_ids], fl[1][selected_ids], fl[2][selected_ids]
    write_ply(filename=selected_ply_path\
              , field_list=fl_selected\
              , field_names=['x', 'y', 'z', 'red', 'green',  'blue', 'class'])
    print("Point cloud selection saved at: \n\t", selected_ply_path)
    
    return fl_selected, selected_ply_path
    
# bx, by = (400, 480), (1090, 1170)
# fl_selected, selected_ply_path = cut_large_ply(bx, by, PLY_FILE_NAME)
