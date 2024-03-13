
import numpy as np
from scipy.spatial import ConvexHull#, convex_hull_plot_2d
from joblib import load


def get_full_facade_from_basepoints(base_points, building_height = 100, points_per_facade = 100, seed=1):
    """
    Generate random locations on the 5 sides facade of four points based building.
    The generated shape is of an  upside down open box.
    bp - the four base points as a 4x3 matrix / np array.
    returns: facade and the eight corners of the generated facade.
    """
    np.random.seed(seed)
    bp     = np.array(base_points)
    num_bp = len(bp)
    #2. points with leveled y
    leveled_bp = np.vstack([bp[:, 0], bp[:, 1].min()*np.ones(len(bp)), bp[:, 2]]).T

    #3 compute order of the points on the base of the building
    hull        = ConvexHull(leveled_bp[:,[0,2]])
    ordered_bp  = leveled_bp[hull.vertices]


    full_facade = []
    #a.facade points of the four latteral walls of the building:
    # for consistency this for loop should be replaced with get_facade_from_ordered_vertices as the rooftop is generated.
    for face_id in range(num_bp):
        #print(face_id, (face_id + 1) % num_bp)
        bp1, bp2 = (ordered_bp[face_id], ordered_bp[(face_id + 1) % num_bp])
        facade_np = get_points_over_segment_by_height(bp1, bp2, height=building_height, num_points=points_per_facade)

        full_facade.append(facade_np)
    
    #b.rooftop facade
    rooftop_bp       = ordered_bp + building_height * np.array([0,1,0])
    rooftop_points   = get_facade_from_ordered_vertices(rooftop_bp, points_per_facade)
    full_facade      = np.vstack(full_facade + [rooftop_points])

    return full_facade, np.vstack([ordered_bp, rooftop_bp])


def generate_angles_with_gmm(num_angles, gmm_path):  
    '''
    Based on trained sklearn GMM model predict new camera angles:
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    returns angles with shape [num_angles, 3]
    '''
    gmm        = load(gmm_path) 
    new_points = gmm.sample(num_angles)[0]
    np.random.seed(1)
    np.random.shuffle(new_points)
    
    #Limit generated samples to +, - 180 otherwise the bounds overflow
    new_angles_capped = np.where(new_points>180, new_points%180, new_points)
    new_angles_capped = np.where(new_angles_capped<-180, new_angles_capped%(-180), new_angles_capped)

    return new_angles_capped


def get_points_over_segment_by_height(p1, p2, height=50, num_points=10):
    """given 2 points in 3d, generate points on a 2d facade above the 2d points
    p = p_1 + \alpha (p2 - p1) + \beta * height 
    with \alpha \beta in (0, 1)
    consider height to be on the second axis.
    """
    
    dv               = p2 - p1 #direction vector
    
    alpha            = np.random.random(num_points)
    beta             = np.random.random(num_points) 

    facade = p1 + alpha.reshape((-1,1)) * dv + height * np.array([0,1,0]) * beta.reshape((-1,1))
    
    
    return facade


def point_on_triangle_uniform(p1, p2, p3, num_points=10):
    """
    Adapted from https://stackoverflow.com/a/68493226/7136493
    """
    dv1 = p2-p1
    dv2 = p3-p1
    alpha             = np.random.random(num_points).reshape((-1, 1))
    beta              = np.random.random(num_points).reshape((-1, 1))

    #if points is in triangle take the small sum else the large sum:
    in_triangle       = (alpha.reshape(-1) + beta.reshape(-1)) <= 1 
    
    #triangle = np.where(in_triangle, dv1 * alpha + dv2 * beta, dv1 * (1-alpha) + dv2 * (1-beta))
    
    small_sum = p1+(dv1 * alpha + dv2 * beta)[in_triangle]
    large_sum = p1+(dv1 * (1-alpha) + dv2 * (1-beta))[~in_triangle]
    
    triangle = np.vstack([small_sum, large_sum])
    
    return triangle

def get_facade_from_ordered_vertices(four_ordered_vertices, num_points=10):
    """
    assuming four counterclockwise orieted vertices generate facade determined by the four corners.
    """
    p1, p2, p3, p4 = four_ordered_vertices

    half_facades = [\
          point_on_triangle_uniform(p2, p3, p1, num_points//2)\
        , point_on_triangle_uniform(p4, p1, p3, num_points//2)\
            ]

    return np.vstack(half_facades)