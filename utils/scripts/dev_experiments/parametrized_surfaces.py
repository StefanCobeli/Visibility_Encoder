import torch
import numpy as np

import open3d as o3d
import seaborn as sns


from utils.scripts.archive.scene_loading_2 import get_highlighed_o3d_locations
from utils.scripts.interest_heuristic_0    import get_o3d_pcd_from_coordinates


    
    

def surface_parametric(a, b, p, c, r, surface_type="circle", lL=None, debugging=False):
    
    a = torch.tensor([float(a)], requires_grad=True)
    b = torch.tensor([float(b)], requires_grad=True)
    
    # Calculate the vector pc
    pc      = c - p
    pc_norm = torch.norm(pc)
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
    if surface_type == "square":
        # Calculate the point on the circle
        if lL is None:
            lL = (r, r)
        l, L = lL
        a = l * (a - .5) * 2
        b = L * (b -.5) * 2 
        if debugging:
            print(a, b)
        square_point = p + r * (a * orient + (b) * v_perpendicular_normalized)
        final_point  = square_point
        
    if surface_type == "sphere":  
        r     = pc_norm
        theta = a * torch.pi     #between 0 and pi
        phi   = b * 2 * torch.pi #between 0 and 2*pi

        x = c[0] + r * torch.sin(theta) * torch.cos(phi)
        y = c[1] + r * torch.sin(theta) * torch.sin(phi)
        z = c[2] + r * torch.cos(theta)

        final_point = c + torch.tensor([r * torch.sin(theta) * torch.cos(phi),0,0])
        final_point = final_point + torch.tensor([0, r * torch.sin(theta) * torch.sin(phi),0])
        final_point = final_point + torch.tensor([0, 0, r * torch.cos(theta)])
        #pass
    
    

    return final_point




# Example usage
p = torch.tensor([10, 10, 10], dtype=torch.float, requires_grad=False)  # Center point
c = torch.tensor([0, 0, 0], dtype=torch.float, requires_grad=False)  # Point on the circle
r = 10  # Radius

# Generate random values for a and b
a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

# orient = torch.tensor([0, 1, 0], dtype=torch.float, requires_grad=False)

# Calculate the point on the circle using parameters a and b
# point_on_circle = surface_parametric(a, b, p, c, r)

surface_type="circle" 
surface_type="sphere" 
# surface_type="square" 
n_points = 1000
# points_on_circle = torch.tensor([circle_parametric(np.random.rand(), np.random.rand(), p, c, r) for i in range(n_points)])
# points_on_circle = [surface_parametric(torch.rand(1, requires_grad=True)\
#                                       , torch.rand(1, requires_grad=True)\
#                                       , p, c, r, surface_type) for i in range(n_points)]

# points_on_circle = [surface_parametric(0\
#                                       , torch.rand(1, requires_grad=True)\
#                                       , p, c, r, surface_type) for i in range(n_points)]

points_on_circle = [surface_parametric(torch.rand(1, requires_grad=True)*.5\
                                      , torch.rand(1, requires_grad=True)\
                                      , p, c, r, surface_type) for i in range(n_points)]

points_on_circle = torch.vstack(points_on_circle).detach().numpy()

# print("Point on the circle:", point_on_circle)

points_on_circ_pcd = get_o3d_pcd_from_coordinates(points_on_circle, [0,1,0])


# circ_pcd     = get_o3d_pcd_from_coordinates(circle_points, [0,1,0]) #green
c_center_pcd = get_highlighed_o3d_locations(p, color=[1,0,0])[0] #red
origin_pcd   = get_highlighed_o3d_locations(c, color=[0,0,1])[0] # blue


# a, b = 1, 1 #angle, distnace to center
# a, b = 0.2, 1
a, b = .5, .5
# a, b = 1, 0
# a, b = 0, 0
# a, b = .25, .25 #angle, distnace to center

new_loc_pcd = get_highlighed_o3d_locations(surface_parametric(a, b, p, c, r, surface_type, debugging=True).detach().numpy(), color=[1,0,1])[0]

new_loc_pcd = get_highlighed_o3d_locations(surface_parametric(a, b, p, c, r, "sphere", debugging=True).detach().numpy(), color=[1,0,1])[0]

# o3d.visualization.draw_geometries([c_center_pcd, origin_pcd, points_on_circ_pcd, new_loc_pcd])

f, l, u, z = [0,0,1], [10,10,0], [0,1,0], 1.6

#CIRCLE
st = "circle"

points_on_circle = torch.vstack([surface_parametric(
                    np.random.rand(1)\
                  , np.random.rand(1)\
                  , p, c, r, st) for i in range(1000)]
).detach().numpy()
points_on_circ_pcd = get_o3d_pcd_from_coordinates(points_on_circle, [0,1,0])

a, b = .25, .75
new_loc_pcd = get_highlighed_o3d_locations(surface_parametric(a, b, p, c, r, st).detach().numpy(), color=[1,0,1])[0]

o3d.visualization.draw_geometries([c_center_pcd, origin_pcd, points_on_circ_pcd, new_loc_pcd], front=f, lookat=l, up=u, zoom=z)


#RECTANGLE
np.random.seed(1)

lL = (1, 2)

st = "square"

points_on_circle = torch.vstack([surface_parametric(
                    np.random.rand(1)\
                  , np.random.rand(1)\
                  , p, c, r, st, lL, False) for i in range(1000)]
).detach().numpy()
points_on_circ_pcd = get_o3d_pcd_from_coordinates(points_on_circle, [0,1,0])

a, b = .25, .75
new_loc_pcd = get_highlighed_o3d_locations(surface_parametric(a, b, p, c, r, st, lL, True).detach().numpy(), color=[1,0,1])[0]

o3d.visualization.draw_geometries([c_center_pcd, origin_pcd, points_on_circ_pcd, new_loc_pcd], front=f, lookat=l, up=u, zoom=z)

#SPHERE:
st = "sphere"
points_on_circle = np.vstack([surface_parametric(torch.rand(1, requires_grad=True)\
                                      , torch.rand(1, requires_grad=True)\
                                      , p, c, r, st) for i in range(n_points)]
)
points_on_circ_pcd = get_o3d_pcd_from_coordinates(points_on_circle, [0,1,0])

a, b = .25, .5
new_loc_pcd = get_highlighed_o3d_locations(surface_parametric(a, b, p, c, r, st).detach().numpy(), color=[1,0,1])[0]

o3d.visualization.draw_geometries([c_center_pcd, origin_pcd, points_on_circ_pcd, new_loc_pcd], front=f, lookat=l, up=u, zoom=z)


#HALF a SPHERE:

st = "sphere"
points_on_circle = np.vstack([surface_parametric(torch.rand(1, requires_grad=True)*.5\
                                      , torch.rand(1, requires_grad=True)\
                                      , p, c, r, st) for i in range(n_points)]
)
points_on_circ_pcd = get_o3d_pcd_from_coordinates(points_on_circle, [0,1,0])

a, b = .25, .5
new_loc_pcd = get_highlighed_o3d_locations(surface_parametric(a, b, p, c, r, st).detach().numpy(), color=[1,0,1])[0]

o3d.visualization.draw_geometries([c_center_pcd, origin_pcd, points_on_circ_pcd, new_loc_pcd], front=f, lookat=l, up=u, zoom=z)

