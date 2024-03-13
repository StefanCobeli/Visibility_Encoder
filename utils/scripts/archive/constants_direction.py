''' 
Contants used by the directional model such as maximum depth and 
width, height, depth ratio.
'''


ground_level_classes    = [0, 4, 5, 6, 7, 10]
immobile_level_classes  = [2, 3]
mobile_level_classes    = [8, 9, 11]
nature_level_classes    = [1, 12]
DEPTHS                  = [64]

COLLISION_DISTANCE      = 7
MAX_STEPS               = 1500  #Maximum steps admitted for scene coverage;
NUM_LOCATIONS           = 2**14 #Num training locations per scene;
NUM_ROTATIONS           = 5


# ins_colors = [[85, 107, 47],    # 0. ground -> OliveDrab
#               [0, 255, 0],      # 1. tree -> Green
#               [255, 165, 0],    # 2. building -> orange
#               [41, 49, 101],    # 3. Walls ->  darkblue
#               [0, 0, 0],        # 4. Bridge -> black
#               [0, 0, 255],      # 5. parking -> blue
#               [255, 0, 255],    # 6. rail -> Magenta
#               [200, 200, 200],  # 7. traffic Roads ->  grey
#               [89, 47, 95],     # 8. Street Furniture  ->  DimGray
#               [255, 0, 0],      # 9. cars -> red
#               [255, 255, 0],    # 10. Footpath  ->  deeppink
#               [0, 255, 255],    # 11. bikes -> cyan
#               [0, 191, 255]     # 12. water ->  skyblue
#               