############  Copied from SensatUrban github repo helper_ply.py:
# https://github.com/QingyongHu/SensatUrban
# https://github.com/QingyongHu/SensatUrban/blob/master/helper_ply.py

############ Also, Copied and adapted from
# SensatUrban github repo tools.py 
# https://github.com/QingyongHu/SensatUrban/blob/master/tool.py


#
#
#      0===============================0
#      |    PLY files reader/writer    |
#      0===============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      function to read/write .ply files
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 10/02/2017
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#
ins_colors = [[85, 107, 47],  # ground -> OliveDrab
    [0, 255, 0],  # tree -> Green
    [255, 165, 0],  # building -> orange
    [41, 49, 101],  # Walls ->  darkblue
    [0, 0, 0],  # Bridge -> black
    [0, 0, 255],  # parking -> blue
    [255, 0, 255],  # rail -> Magenta
    [200, 200, 200],  # traffic Roads ->  grey
    [89, 47, 95],  # Street Furniture  ->  DimGray
    [255, 0, 0],  # cars -> red
    [255, 255, 0],  # Footpath  ->  deeppink
    [0, 255, 255],  # bikes -> cyan
    [0, 191, 255]  # water ->  skyblue
]
# ins_colors = [[round(r/257, 2), round(g/257, 2), round(b/257, 2)] for (r, g, b) in ins_colors]
ins_colors = [(r, g, b) for (r, g, b) in ins_colors]

ins_names  = ["ground",            "tree",    "building", "Walls"
            , "Bridge",            "parking", "rail",     "traffic Roads"
            , "Street Furniture ", "cars",    "Footpath", "bikes"
            ,  "water"
]
ins_dict       = dict(zip([tuple(c) for c in ins_colors], ins_names))
ins_rev_dict   = dict(zip(ins_names, [tuple(c) for c in ins_colors]))

from collections import OrderedDict
ins_dict     = OrderedDict(ins_dict)#.values()
ins_rev_dict = OrderedDict(ins_rev_dict)

def sem_name_to_color(sem_name):
    r, g, b = ins_rev_dict[sem_name]
    sem_color = tuple([round(r, 2), round(g, 2), round(b, 2)])
    return sem_color

def sem_color_to_name(sem_color, names_and_ids=False):
    r, g, b = sem_color
    sem_id  = tuple([round(r, 2), round(g, 2), round(b, 2)])
    label_name = ins_dict[sem_id]
    label_id   = ins_names.index(label_name)
    if names_and_ids:
        return label_name, label_id #ins_dict[sem_id], ins_names.index(ins_dict[sem_id])
    else:
        return label_id
# Basic libs
import numpy as np
import sys


# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


############ Copied and adapted from
# tools.py 
# https://github.com/QingyongHu/SensatUrban/blob/master/tool.py

from os.path import join, exists, dirname, abspath
import numpy as np
import colorsys, random, os, sys
import open3d as o3d
# from helper_ply import read_ply, write_ply

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
# import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigSensatUrban:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.2  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 14  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_SensatUrban'
    saving = True
    saving_path = None


class DataProcessing:

    @staticmethod
    def get_num_class_from_label(labels, total_class):
        num_pts_per_class = np.zeros(total_class, dtype=np.int32)
        # original class distribution
        val_list, counts = np.unique(labels, return_counts=True)
        for idx, val in enumerate(val_list):
            num_pts_per_class[val] += counts[idx]
        # for idx, nums in enumerate(num_pts_per_class):
        #     print(idx, ':', nums)
        return num_pts_per_class

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    # @staticmethod
    # def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    #     """
    #     CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    #     :param points: (N, 3) matrix of input points
    #     :param features: optional (N, d) matrix of features (floating number)
    #     :param labels: optional (N,) matrix of integer labels
    #     :param grid_size: parameter defining the size of grid voxels
    #     :param verbose: 1 to display
    #     :return: sub_sampled points, with features and/or labels depending of the input
    #     """

    #     if (features is None) and (labels is None):
    #         return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
    #     elif labels is None:
    #         return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
    #     elif features is None:
    #         return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    #     else:
    #         return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
    #                                        verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def read_ply_data(path, with_rgb=True, with_label=True, with_interest_label=False):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']
            if with_interest_label:
                interest = data[f"interest"]
                return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8), interest.astype(np.uint8)
            return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        elif not with_rgb and with_label:
            labels = data['class']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)

    @staticmethod
    def random_sub_sampling(points, features=None, labels=None, sub_ratio=10, verbose=0):
        num_input = np.shape(points)[0]
        num_output = num_input // sub_ratio
        idx = np.random.choice(num_input, num_output)
        if (features is None) and (labels is None):
            return points[idx]
        elif labels is None:
            return points[idx], features[idx]
        elif features is None:
            return points[idx], labels[idx]
        else:
            return points[idx], features[idx], labels[idx]

    @staticmethod
    def get_class_weights(num_per_class, name='sqrt'):
        # # pre-calculate the number of points in each category
        frequency = num_per_class / float(sum(num_per_class))
        if name == 'sqrt' or name == 'lovas':
            ce_label_weight = 1 / np.sqrt(frequency)
        elif name == 'wce':
            ce_label_weight = 1 / (frequency + 0.02)
        else:
            raise ValueError('Only support sqrt and wce')
        return np.expand_dims(ce_label_weight, axis=0)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb, return_pc=False):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        if return_pc:
            return pc

        #o3d.geometry.PointCloud.estimate_normals(pc)
        o3d.visualization.draw_geometries([pc], width=1000, height=1000)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None, return_pc=True):
        # only visualize a number of points to save memory
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            # ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)
            ins_colors = [[85, 107, 47],  # ground -> OliveDrab
                          [0, 255, 0],  # tree -> Green
                          [255, 165, 0],  # building -> orange
                          [41, 49, 101],  # Walls ->  darkblue
                          [0, 0, 0],  # Bridge -> black
                          [0, 0, 255],  # parking -> blue
                          [255, 0, 255],  # rail -> Magenta
                          [200, 200, 200],  # traffic Roads ->  grey
                          [89, 47, 95],  # Street Furniture  ->  DimGray
                          [255, 0, 0],  # cars -> red
                          [255, 255, 0],  # Footpath  ->  deeppink
                          [0, 255, 255],  # bikes -> cyan
                          [0, 191, 255]  # water ->  skyblue
                          ]
        colors_array = []
        for psi in pc_sem_ins:
            colors_array.append(ins_colors[psi])
        colors_array = np.array(colors_array)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyz)
        pc.colors = o3d.utility.Vector3dVector(colors_array / 255.)
        if return_pc:
            return pc

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            # xmin = np.min(valid_xyz[:, 0]);
            # xmax = np.max(valid_xyz[:, 0])
            # ymin = np.min(valid_xyz[:, 1]);
            # ymax = np.max(valid_xyz[:, 1])
            # zmin = np.min(valid_xyz[:, 2]);
            # zmax = np.max(valid_xyz[:, 2])
            # sem_ins_bbox.append(
            #     [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        pc = Plot.draw_pc(Y_semins, return_pc)

        if return_pc:
            return pc

        return Y_semins
    
    @staticmethod
    def save_ply_o3d(data, save_name):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
        if np.shape(data)[1] == 3:
            o3d.io.write_point_cloud(save_name, pcd)
        elif np.shape(data)[1] == 6:
            if np.max(data[:, 3:6]) > 20:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.)
            else:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
            o3d.io.write_point_cloud(save_name, pcd)
        return

    @staticmethod
    def get_rgb_from_urban_labels(pc_sem_ins, plot_colors=None):
        # only visualize a number of points to save memory
        if plot_colors is not None: #plot_colors are custom for each point
            ins_colors = plot_colors
        else:
            # ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)
            ins_colors = [[85, 107, 47],  # ground -> OliveDrab
                        [0, 255, 0],  # tree -> Green
                        [255, 165, 0],  # building -> orange
                        [41, 49, 101],  # Walls ->  darkblue
                        [0, 0, 0],  # Bridge -> black
                        [0, 0, 255],  # parking -> blue
                        [255, 0, 255],  # rail -> Magenta
                        [200, 200, 200],  # traffic Roads ->  grey
                        [89, 47, 95],  # Street Furniture  ->  DimGray
                        [255, 0, 0],  # cars -> red
                        [255, 255, 0],  # Footpath  ->  deeppink
                        [0, 255, 255],  # bikes -> cyan
                        [0, 191, 255]  # water ->  skyblue
                    ]

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                    # tp = ins_colors[id] #Bug
                else:
                    tp = ins_colors[id]
                    # tp = ins_colors[semins]# Bug

            Y_colors[valid_ind] = tp
        return Y_colors

            ### bbox
            # valid_xyz = pc_xyz[valid_ind]

            # xmin = np.min(valid_xyz[:, 0]);
            # xmax = np.max(valid_xyz[:, 0])
            # ymin = np.min(valid_xyz[:, 1]);
            # ymax = np.max(valid_xyz[:, 1])
            # zmin = np.min(valid_xyz[:, 2]);
            # zmax = np.max(valid_xyz[:, 2])
            # sem_ins_bbox.append(
            #     [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        #Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        #Plot.draw_pc(Y_semins)
        #return Y_semins