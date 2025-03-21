
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



'''
Parametic Surface generation
'''
import torch

def surface_parametric(a, b, p, c, r, surface_type="circle", debugging=False):
    """
    a, b - parameters between 0, 1 (can be random)
    p, c - 3d points defining the surface - prefereably torch tensors
    r, surface_type="circle", lL=None, debugging=False
    returns:
    final_point - 3d point correspondig to the parameters ab, on the surface pc.
    """
    a = torch.tensor([float(a)], requires_grad=True)
    b = torch.tensor([float(b)], requires_grad=True)
    p = torch.tensor(np.array(p, dtype=float), requires_grad=True)
    c = torch.tensor(np.array(c, dtype=float), requires_grad=True)
    # Calculate the vector pc
    pc      = c - p
    pc_norm = torch.norm(pc)
    pc      = pc / pc_norm
    orient  = torch.tensor([1,1, (- pc[0] - pc[1]) / pc[2]])
    orient  = orient / torch.norm(orient)

    # Normalize pc to get the direction
    pc_normalized = pc / pc_norm

    # Find a vector perpendicular to pc
    v_perpendicular = torch.cross(pc_normalized, orient)

    # Normalize v_perpendicular
    v_perpendicular_normalized = v_perpendicular / torch.norm(v_perpendicular)
    
    if surface_type =="circle":
        # Calculate angle theta
        theta = 2 * np.pi * a
        # Calculate the point on the circle
        circle_point = p + r * (torch.cos(theta) * orient + torch.sin(theta) * v_perpendicular_normalized)
        # Interpolate between p and the circle_point using parameter b
        final_point = (1 - b) * p + b * circle_point
        if debugging:
            print(pc, a,b,orient, pc_norm, theta, circle_point, final_point)
    if surface_type == "square":
        # Calculate the point on the circle
        # if lL is None:
        #     lL = (r, r); l,L=lL
        l, L = r
        a = l * a#  * 2
        b = L * b#  * 2 
        # p_min = p - l * orient - L * v_perpendicular_normalized
        # square_point = p + r * (a * orient + (b) * v_perpendicular_normalized)
        square_point = p + a * v_perpendicular_normalized + b * orient 
        final_point  = square_point
        if debugging:
            print(pc, a,b,orient, pc_norm, theta, circle_point, final_point)
        
    if surface_type == "sphere":  
        #r     = pc_norm
        theta = a * torch.pi     #between 0 and pi
        phi   = b * 2 * torch.pi #between 0 and 2*pi

        x = c[0] + r * torch.sin(theta) * torch.cos(phi)
        y = c[1] + r * torch.sin(theta) * torch.sin(phi)
        z = c[2] + r * torch.cos(theta)

        final_point = c + torch.tensor([r * torch.sin(theta) * torch.cos(phi),0,0])
        final_point = final_point + torch.tensor([0, r * torch.sin(theta) * torch.sin(phi),0])
        final_point = final_point + torch.tensor([0, 0, r * torch.cos(theta)])
       
        if debugging:
            print(pc, a,b,orient, pc_norm, theta, circle_point, final_point)
    

    return final_point.to(torch.float32)


# Experiment generate facedes in rubik cube style tiles.
# moved from  `1_encoder_experiment_training_density_requirements.ipynb` - 01.21.2025 

def sort_polygon_points(points):
    """
    Sort a list of 3D points to ensure they are in a clockwise order 
    around the polygon in the XY plane.

    Parameters:
    ----------
    points : list of [x, y, z]
        A list of 3D points defining the polygon vertices.

    Returns:
    -------
    sorted_points : list of [x, y, z]
        The input points sorted in clockwise order.
    """
    # Compute the centroid of the polygon
    centroid = np.mean(points, axis=0)

    # Compute angles of each point relative to the centroid in the XY plane
    angles = [np.arctan2(p[1] - centroid[1], p[0] - centroid[0]) for p in points]

    # Sort points by angle (clockwise order)
    sorted_indices = np.argsort(angles)
    sorted_points = [points[i] for i in sorted_indices]
    # print("Unsorted array:", sorted_points[::-1])
    # print("Sorted array:", np.array(sorted_points[::-1]))
    return np.array(sorted_points[::-1])


# Moved from 2_Buildings_to_Exterior_Use_Case and 1_encoder_experiment_training_density_requirements - 03.10.2025
from scipy.spatial.transform import Rotation as R
def compute_perpendicular_orientation(p1, p2):
    # Convert to NumPy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Compute direction vector (p1 -> p2)
    direction = p1 - p2 + np.ones(3) * 1e-4
#     print(p1, p2)
    direction /= np.linalg.norm(direction)  # Normalize

    # Choose a reference up vector (default is [0,1,0])
    world_up = np.array([0, 1, 0])

    # Handle collinearity by switching to another vector if needed
    if abs(np.dot(direction, world_up)) > 0.99:  # Almost parallel
        world_up = np.array([1, 0, 0])

    # Compute a perpendicular right vector
    right = np.cross(world_up, direction)
    right /= np.linalg.norm(right)  # Normalize

    # Compute a valid up vector to maintain a proper coordinate system
    up = np.cross(direction, right)
    up /= np.linalg.norm(up)  # Normalize

    # Construct rotation matrix (basis: right, up, direction)
    rotation_matrix = np.column_stack((right, up, direction))
    

    # Apply a 90-degree rotation around X-axis to match Three.js
    correction_matrix = R.from_euler('y', 90, degrees=True).as_matrix()
#     correction_matrix = R.from_euler('x', -90, degrees=True).as_matrix()
    corrected_matrix = correction_matrix @ rotation_matrix  # Adjust frame

    # Convert to Euler angles (YXZ convention)
    euler_angles = R.from_matrix(corrected_matrix).as_euler('yxz', degrees=True)
    euler_angles = euler_angles[[1, 0, 2]]

#     return euler_angles#
    return np.round(euler_angles, 5 )

def generate_vertical_squares(points, n_width, n_height, n_samples, natural_height=True, verbose=True):
    """
    Parameters:
    ----------
    points : list of [x, y, z]
        Four 3D points defining the polygon vertices.
    n_width : int
        Number of squares to divide the shortest edge into (horizontal division).
    n_height : int
        Number of squares to stack vertically (vertical division).
    n_samples : int
        Number of random points to generate inside each square.
    natural_height : boolean
        If n_height is to be treated as the actual height of the building, not as how many squares should be stacked up.
        if natural_height, then n_height shoudd become n_height // square_side

    Returns:
    -------
    square_centers : list of [x, y, z] - length is multiple of 4 / number of sides.
        Centers of all generated squares. #ordered based on each side [side1_c1, side1_c2,...side2_c1,...,side4_cn]
    square_samples : list of lists of [x, y, z]
        Random points within each square.
    square_side : float
        The length of each square's side.
        
    Example:
    --------
    >>> points = [
    ...     [0, 0, 0],  # Bottom-left corner
    ...     [1, 0, 0],  # Bottom-right corner
    ...     [1, 1, 0],  # Top-right corner
    ...     [0, 1, 0]   # Top-left corner
    ... ]
    # Or example with unordered points - Sorting perforemed using `sort_polygon_points`
    points = [[1208.93435038,   26.89583837,  471.78952441],
     [1235.56810426,   26.89584999,  482.85764003],
     [1227.46644624,   26.89580719,  427.19482581],
     [1254.10020012,   26.89581881,  438.26294142]
     ]
    >>> n_width = 5
    >>> n_height = 3
    >>> n_samples = 10
    >>> centers, samples, side_length = generate_vertical_squares(points, n_width, n_height, n_samples)
    >>> print(centers)
    >>> print(samples)
    >>> print(side_length)
    """
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def compute_normal(polygon_points):
        # Compute the normal vector of the polygon (assume it's planar)
        v1 = np.array(polygon_points[1]) - np.array(polygon_points[0])
        v2 = np.array(polygon_points[2]) - np.array(polygon_points[0])
        # normal = np.cross(v2, v1)#
        normal = np.cross(v1, v2)
        normed_normal = normal / np.linalg.norm(normal)
        normed_normal_positive_height = np.array([normed_normal[0],
                                                np.abs(normed_normal[1]),
                                                normed_normal[2]])
        return normed_normal_positive_height
    
    # print(f"received points:", np.round(points, 2))
    points = sort_polygon_points(points)
    # print(f"reordered points:", np.round(points, 2))
    
    segments = [(points[i], points[(i + 1) % 4]) for i in range(4)]
    lengths = [distance(p1, p2) for p1, p2 in segments]

    shortest_length = min(lengths)
    square_side = shortest_length / n_width

    if natural_height:
        n_height = int(n_height // square_side) + 1
        
    if verbose:
        print(f"For the height {n_height}, there will be generated {n_height//square_side} square tiles of size {square_side}.")

    # Compute the normal vector of the polygon
    normal_vector = compute_normal(points)

    square_centers = [] # list of tile centers
    square_samples = [] # list of lists for each center we have n_samples samples
    side_ids       = [] # which one of the 4 sides does each center belong to 

    #centers are list with the order [side1_c1, side1_c2,...side2_c1,...,side4_cn]
    for (sid, ((p1, p2), length)) in enumerate(zip(segments, lengths)):
        segment_vector  = np.array(p2) - np.array(p1)
        segment_length  = np.linalg.norm(segment_vector)
        unit_vector     = segment_vector / segment_length
        vertical_vector = normal_vector  # Vertical vector is the polygon's normal

        num_squares_along = int((segment_length * 1.1)  // square_side) #Make sure the segment_length is not divisible by square_side, otherwise you get one square missing on the side.
        
        for i in range(num_squares_along):
            for j in range(n_height):
                side_ids.append(sid)
                # Compute the center in the vertical plane perpendicular to the polygon
                center = (
                    np.array(p1) +
                    (i + 0.5) * square_side * unit_vector +
                    (j + 0.5) * square_side * vertical_vector
                )
                square_centers.append(center)

                # Generate random points within the square
                random_offsets = np.random.rand(n_samples, 2) - 0.5
                random_points = (
                    center +
                    square_side * (random_offsets[:, 0][:, None] * unit_vector + random_offsets[:, 1][:, None] * vertical_vector)
                )
                square_samples.append(random_points.tolist())

    return square_centers, square_samples, square_side, side_ids


def draw_facade_centers_and_tiles_in_o3d(points, centers, samples):
    import open3d as o3d

    # Draw centers
    pcd_centers = o3d.geometry.PointCloud()
    pcd_centers.points = o3d.utility.Vector3dVector(centers)
    pcd_centers.colors = o3d.utility.Vector3dVector(np.repeat([[0,0,1]], len(centers), axis=0))


    # Draw basis / original points
    pcd_corners = o3d.geometry.PointCloud()
    pcd_corners.points = o3d.utility.Vector3dVector(points)
    pcd_corners.colors = o3d.utility.Vector3dVector(np.repeat([[1,0,0]], len(points), axis=0))

    # Draw tile sampled points.
    pcd_samples = o3d.geometry.PointCloud()
    pcd_samples.points = o3d.utility.Vector3dVector(np.vstack(samples))
    np.random.seed(1)
    n_samples = len(np.vstack(samples)) // len(centers)
    np_random_tile_colors = np.repeat(np.random.rand(len(centers), 3), n_samples, axis=0)
    pcd_samples.colors = o3d.utility.Vector3dVector(np_random_tile_colors)

    o3d.visualization.draw_geometries([pcd_centers, pcd_corners])
    o3d.visualization.draw_geometries([pcd_centers, pcd_corners, pcd_samples])