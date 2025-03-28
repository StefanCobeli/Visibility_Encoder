
from textwrap import indent
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from utils.scripts.architectures.train_location_encoder import *
from utils.test_location_encoder                        import *
from utils.view_impact                                  import remove_builiding_and_retrain_model, reset_encoder_weights
from utils.gradient_walk_utils                          import query_locations, query_locations_on_surface#, load_model_from_info_dict_path
import time

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app)


@app.route('/query_plane_locations', methods=["POST", "GET"])
def query_plane_locations_page():
    '''
    http://127.0.0.1:5000/query_locations_plane
    or in command line 
    Semantics query on ["building", "water", "road", "sidewalk", "surface", "tree", "sky"]
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/query_locations/query_full_semantics.json "http://127.0.0.1:5000/query_plane_locations"

    Perception query on ["greeness", "openness", "imageability", "enclosure", "walkability", "serenity"]
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/query_locations/query_perception.json "http://127.0.0.1:5000/query_plane_locations"

    Original examples    
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/query_locations/query_plane.json "http://127.0.0.1:5000/query_plane_locations"
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/query_locations/query_plane_directions.json "http://127.0.0.1:5000/query_plane_locations"
    '''
    seed = np.random.randint(10**6)
    # print("Random seed:", seed)
    # seed = 1 # the random seed either is the same always or is never set
    # si   = np.ones(4) * .005 # search intervals
    lt   =.0001      # loss threshould under which the optimization is stopped.
    lr   = 5 * 1e-2 #learning rate for optimization on the plane
    # lr   = 5*1e-1 #learning rate for optimization on the plane
    ms   = 50     # maximum optimization steps
    
    try:
        data                 = request.json
        query_df             = pd.DataFrame(data)

        print(query_df["f_xyz"].values[0], type(query_df["f_xyz"].values[0]), type(query_df["f_xyz"].values[0]) is dict)
        print(list(query_df["f_xyz"].values[0].values())[0], type(list(query_df["f_xyz"].values[0].values())[0]))#Custom perception: Check if each dictionary entry is itself a dictionary.
        # return query_df.to_json(orient="records", indent=4)
        # return query_df.to_json(orient="records", indent=4)
        
        p = torch.tensor([float(x) for x in query_df["point_on_plane"].values[0]]).to(torch.float32)
        if "point_plus_normal" in query_df:
            c = torch.tensor([float(x) for x in query_df["point_plus_normal"].values[0]]).to(torch.float32)
            #optional_directions = None
        else:
            #c = None
            direction_1 = ([float(x) for x in query_df["direction_1"].values[0]])
            direction_2 = ([float(x) for x in query_df["direction_2"].values[0]])
            c = (direction_1, direction_2)
            #print(c)
        r = torch.tensor([float(x) for x in query_df["r"].values[0]])
        
        if type(query_df["f_xyz"].values[0]) is dict or type(query_df["f_xyz"].values[0]) is list:
            desired_distribution = query_df["f_xyz"].values[0] # moved decision of parsing query_df["f_xyz"] to query_locations_on_surface
       
        si   = np.ones(len(query_df["f_xyz"])) * .05 #.5 # search intervals

        surface_basis        = (p, c, r); surface_type = "square"
        num_locations        = int(query_df["num_locations"].values[0])
        # if "seed" in query_df:
        #     seed = int(query_df["seed"])
    except Exception as e:
        print(e)
        print(f"Invalid JSON sent in the request - two examples of query locations on a plane are is in:")
        print("\t ./utils/assets/query_locations/query_plane.json")
        print("\t ./utils/assets/query_locations/query_plane_directions.json")
        return jsonify([{"":"Invalid JSON"}])

    al_df    = query_locations_on_surface(desired_distribution, surface_basis, surface_type\
       , num_locations=num_locations, search_intervals=si, lt=lt, lrate=lr, max_steps = ms, seed=seed)
    # print(al_df.x.max(axis=0), al_df.x.min(axis=0), al_df.z.max(axis=0), al_df.z.min(axis=0))
    # al_df.to_csv("./utils/assets/query_locations/queried_locations_plane.csv", index=False)
    al_df.to_json("./utils/assets/query_locations/queried_locations_plane.csv", orient="records", indent=4)

    return al_df.to_json(orient="records", indent=4)


@app.route('/query_locations', methods=["POST", "GET"])
def query_locations_page():
    '''
    http://127.0.0.1:5000/query_locations
    or in command line 
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/query_locations/location.json "http://127.0.0.1:5000/query_locations"
    '''
    # seed = 11  # Returns always the same results. Comment this to get different locations.  
    seed             = np.random.randint(10**6) # Don't set seeds on the locations.json to get random locations
    search_intervals = np.ones(4) * .01 #.2 search_intervals=np.ones(4)*.01, lt=0.01, at=20, max_steps=100)
    at = 100  # acceptable factor loss; if loss is at least at * lt then consider the location
    lt = .01 # loss threshold. Don't perform additional steps if the loss is already lt.
    max_steps = 75 #100                    #100

    try:
        data                 = request.json
        query_df             = pd.DataFrame(data)
        query_df["f_xyz"].values[0]["miscellaneous"] = 0

        desired_distribution = query_df["f_xyz"].values[0] 
        desired_keys         = [str(x) for x in query_df["f_xyz"].values[0].keys()]
        desired_distribution = [float(x) for x in query_df["f_xyz"].values[0].values()]
        num_locations        = int(query_df["num_locations"].values[0])
        print(f"The received query is #{num_locations} locations:\n\t{desired_keys} : {desired_distribution}")
        # print("query_df is ", query_df)
        default_label_order  = ['building', 'water', 'road', 'sidewalk', 'surface', 'tree', 'sky', 'miscellaneous']
        reordering_query_ids = [desired_keys.index(s) for s in default_label_order]
        desired_keys         = [desired_keys[i] for i in reordering_query_ids]
        desired_distribution = [desired_distribution[i] for i in reordering_query_ids]
        print(f"Reordered query is #{num_locations} locations:\n\t{desired_keys} : {desired_distribution}")
        # print("query_df is ", query_df)
        search_intervals     = np.ones(len(query_df["f_xyz"])) * .05
        #if "seed" in query_df:
        #    seed = int(query_df["seed"])
    except Exception as e:
        print(e)
        print(f"Invalid JSON sent in the request - an example of query locations file is in:")
        print("\t ./utils/assets/query_locations/location.json")
        return jsonify([{"":"Invalid JSON"}])


    al_df = query_locations(desired_distribution, num_locations, search_intervals, lt, at, max_steps, seed)
    al_df 
    #al_df = query_locations(desired_distribution, num_locations, seed)
    al_df.to_csv("./utils/assets/query_locations/query_location.csv", index=False)

    return al_df.to_json(orient="records", indent=4)


@app.route('/remove_building', methods=["POST", "GET"])
def remove_building_page():
    '''
    Remove building and retrain model based on received recomputed locations:
    test in browser using:
    http://127.0.0.1:5000/remove_building
    or in command line 
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/removed_buildings/removedBuilding.json "http://127.0.0.1:5000/remove_building"
    '''
    reset_encoder_weights()
    removed_path  = "./utils/assets/removed_buildings/removedBuilding.csv"
    try:
        data          = request.json
        print("Saving df!")
        removed_df    = pd.DataFrame(data)
        removed_df.to_csv(removed_path, index=False)
        test_losses_history, rm_losses_history = remove_builiding_and_retrain_model()
        print("Trained!")

    except Exception as e:
        print(e)
        print(f"Invalid JSON sent in the request - an example removed locations file is in:\n\t{removed_path.strip('.csv')}.json")
        rm_losses_history = ["np.inf"]; test_losses_history = ["np.inf"]

    print(rm_losses_history, test_losses_history)
    return jsonify([{"removed losses":np.vstack(rm_losses_history).tolist(), "test losses": np.vstack(test_losses_history).tolist()}])


# DISCRETE: Split facade into tiles and display scalar field / random locations within each tile.
@app.route('/predict_facade_from_base_points_as_tiles', methods=["POST", "GET"])
def predict_facade_from_base_points_as_tiles_page():
    '''   
    ///receives argumets - data: basePoints, bh, ppf and 
    http://127.0.0.1:5000/predict_facade_from_base_points_as_tiles
    or in command line:
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/new_buildings/base_points_example.json "http://127.0.0.1:5000/predict_facade_from_base_points_as_tiles"
    '''
    base_points_name = request.args.get('bpn', 'unnamed_points') # base points name #"base_points_example"
    base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')

    try:
        data  = request.json
        print("Received data was:\n\t", data)
        bp_df = pd.DataFrame(data)#.to_dict(orient="records")

        start_time = time.time()
        #1. Read base points
        
        bp  = bp_df.values
        print("Received base points were:", bp, "\n")

        #Performance logs:
        total_time = time.time() - start_time
        print(f"\t prediction time: total - {total_time:.2f}s; " )
            #  f"per location - {total_time/facade_df.shape[0]:.7f}\n")
        bh  = int(request.args.get('bh', '50'))   #buiding height
        ppf = int(request.args.get('ppf', '250')) #points per facade side
        print(f"Received height was {bh},\n ppf\t {ppf}, \n ")


        from utils.geometry_utils import generate_vertical_squares

        n_width = 20#10#15 #number of tiles on the thinnest side.
        n_height = bh#20 #the height of the builidng, not how many tiles will the facade have
        n_samples = 6 #make this always divisible to 6 due to interface hard coding #5#100 #sampels per tile
        points    = bp

        #centers, samples, side_length = generate_vertical_squares(points, n_width, n_height, n_samples) #used in get_facade_predictions_as_tiles
        info_dict_path = "./utils/assets/data/semantics/models/training_info_100.json"
        
        from utils.test_location_encoder import get_facade_predictions_as_tiles
        facade_dict = get_facade_predictions_as_tiles(points, n_height, info_dict_path, n_width, n_samples)[0]

        print("predicted dict keys: ", list(facade_dict.keys()))

    except:
        print("Empty JSON sent in the request - Using the Example base points")
        base_points_name = "base_points_example"
        base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')
        bp_df            = pd.read_csv(base_points_path, header=None)


    #Mimicking the random locations display method below  (predict_facade_from_base_points_page)
    # facade_dict = {}
    # ['building', 'water', 'road ', 'sidewalk', 'surface', 'tree', 'sky', 'miscellaneous']
    #Which prediction should be the first column. By default is 0
    # first_prediction_id = 0 #Buildings
    # first_prediction_id = 1 #Water
    # first_prediction_id = 2 #Road
    # first_prediction_id = 3 #sidewalk
    # first_prediction_id = 4 #surface
    first_prediction_id = 5 #Tree
    # first_prediction_id = 6 #Sky
    # first_prediction_id = 7 #miscellaneous
    for fd in facade_dict:
        p1 = fd["predictions"][first_prediction_id]
        pb = fd["predictions"][0]
        fd["predictions"][first_prediction_id] = pb
        fd["predictions"][0] = p1
    
    ##Normalize predictions to maximum of the predicted class:

    print("\nNot Jsonified response sample: ", facade_dict[:2])
    response = jsonify(facade_dict)
    print("Jsonified response:", response)
    # response.headers.add('Access-Control-Allow-Origin', '*')

    # facade_dict = [{"camera_coordinates":[fr["x"],fr["y"],fr["z"],fr["xh"],fr["yh"],fr["zh"]]\
    #     , "predictions": fr["predictions"] } for fr in facade_records]
    # response = jsonify(facade_dict)

    return response

# Split facade into tiles and display scalar field / random locations within each tile.
@app.route('/predict_facade_from_base_points_as_continous_tiles', methods=["POST", "GET"])
def predict_facade_from_base_points_as_continous_tiles_page():
    '''   
    ///receives argumets - data: basePoints, bh, ppf and 
    http://127.0.0.1:5000/predict_facade_from_base_points_as_continous_tiles
    or in command line:
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/new_buildings/base_points_example.json "http://127.0.0.1:5000/predict_facade_from_base_points_as_continous_tiles"
    '''
    base_points_name = request.args.get('bpn', 'unnamed_points') # base points name #"base_points_example"
    base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')

    try:
        data  = request.json
        print("Received data was:\n\t", data)
        bp_df = pd.DataFrame(data)#.to_dict(orient="records")

        start_time = time.time()
        #1. Read base points
        
        bp  = bp_df.values
        print("Received base points were:", bp, "\n")

        #Performance logs:
        total_time = time.time() - start_time
        print(f"\t prediction time: total - {total_time:.2f}s; " )
            #  f"per location - {total_time/facade_df.shape[0]:.7f}\n")
        bh  = int(request.args.get('bh', '50'))   #buiding height
        ppf = int(request.args.get('ppf', '250')) #points per facade side #DEPRECATED
        print(f"Received height was {bh},\n ppf\t {ppf}, \n ")


        from utils.geometry_utils import generate_vertical_squares

        n_width = 10#20#10#15 #number of tiles on the thinnest side.
        n_height = bh#20 #the height of the builidng, not how many tiles will the facade have
        n_samples = 5 #make this always divisible to 6 due to interface hard coding #5#100 #sampels per tile
        points    = bp

        #centers, samples, side_length = generate_vertical_squares(points, n_width, n_height, n_samples) #used in get_facade_predictions_as_tiles
        info_dict_path = "./utils/assets/data/semantics/models/training_info_100.json"
        
        discretization_type = "linear"
        # discretization_type = "exponential"
        # color_normalization   = "max"
        # color_normalization   = None
        color_normalization = {'building': 0.9, 'water': 0.4, 'road': 0.1, 'sidewalk': 0.05, 'surface': 0.15, 'tree': 0.85, 'sky': 0.85, 'miscellaneous': 0.01}#maximums_handpicked


        from utils.test_location_encoder import get_facade_predictions_as_tiles
        facade_dict = get_facade_predictions_as_tiles(points, n_height, info_dict_path, n_width, n_samples, discretization_type, verbose=True, color_normalization=color_normalization)[1]

        # print("predicted dict keys: ", list(facade_dict.keys()))

    except:
        print("Empty JSON sent in the request - Using the Example base points")
        base_points_name = "base_points_example"
        base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')
        bp_df            = pd.read_csv(base_points_path, header=None)


    #Mimicking the random locations display method below  (predict_facade_from_base_points_page)
    # facade_dict = {}
    # ['building', 'water', 'road ', 'sidewalk', 'surface', 'tree', 'sky', 'miscellaneous']
    #Which prediction should be the first column. By default is 0
    # first_prediction_id = 0 #Buildings
    # first_prediction_id = 1 #Water
    # first_prediction_id = 2 #Road
    # first_prediction_id = 3 #sidewalk
    # first_prediction_id = 4 #surface
    # first_prediction_id = 5 #Tree
    # first_prediction_id = 6 #Sky
    # first_prediction_id = 7 #miscellaneous
    # for fd in facade_dict:
    #     p1 = fd["predictions"][first_prediction_id]
    #     pb = fd["predictions"][0]
    #     fd["predictions"][first_prediction_id] = pb
    #     fd["predictions"][0] = p1
    
    ##Normalize predictions to maximum of the predicted class:

    print("\nNot Jsonified response sample: ", facade_dict[:2])
    response = jsonify(facade_dict)
    print("Jsonified response:", response)
    # response.headers.add('Access-Control-Allow-Origin', '*')

    # facade_dict = [{"camera_coordinates":[fr["x"],fr["y"],fr["z"],fr["xh"],fr["yh"],fr["zh"]]\
    #     , "predictions": fr["predictions"] } for fr in facade_records]
    # response = jsonify(facade_dict)

    return response

# Create locations on facade of the building - random points on facade. Not ordered tiles.
@app.route('/predict_facade_from_base_points', methods=["POST", "GET"])
def predict_facade_from_base_points_page():
    '''
    generated predictions and inputs on facade from base points 
    test in browser using:
    http://127.0.0.1:5000/predict_facade_from_base_points?bh=50&ppf=250
    or in command line:
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/new_buildings/base_points_example.json "http://127.0.0.1:5000/predict_facade_from_base_points?bh=10&ppf=150"
    '''

    base_points_name = request.args.get('bpn', 'unnamed_points') # base points name #"base_points_example"
    base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')

    try:
        data  = request.json
        print(data)
        bp_df = pd.DataFrame(data)#.to_dict(orient="records")
    except:
        print("Empty JSON sent in the request - Using the Example base points")
        base_points_name = "base_points_example"
        base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')
        bp_df            = pd.read_csv(base_points_path, header=None)

    start_time = time.time()
    #1. Read base points
    
    bp  = bp_df.values
    print("Received base points were:", bp, "\n")
    bh  = int(request.args.get('bh', '50'))   #buiding height
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    ppf = int(request.args.get('ppf', '250')) #points per facade side
    nmt = request.args.get('nmt', 'percentages') #normalization type
    print(f"Received height was {bh},\n ppf\t {ppf}, \n ")

    #2. Predict facade valeus for given inputs
    facade_df = predict_facade_from_base_points(bp, bh, ppf, nmt, bs)

    #Performance logs:
    total_time = time.time() - start_time
    print("Buidling dimensions:", base_points_path.split("/")[-1][:-4]) 
    xM, yM, zM = facade_df.max()[["x","y","z"]]
    xm, ym, zm = facade_df.min()[["x","y","z"]]
    print(f"\t dx {xM-xm:.1f}; dy {yM-ym:.1f}; dz {zM-zm:.1f}")
    print(f"\t #points: {facade_df.shape[0]:,}")
    print(f"\t prediction time: total - {total_time:.2f}s; per location - {total_time/facade_df.shape[0]:.7f}\n")

    facade_records = facade_df.to_dict(orient="records")
    facade_dict = [{"camera_coordinates":[fr["x"],fr["y"],fr["z"],fr["xh"],fr["yh"],fr["zh"]]\
        , "predictions": fr["predictions"] } for fr in facade_records]
    response = jsonify(facade_dict)
    print("Jsonified response:", response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/test_encoder_on_current_position", methods=['POST', "GET"])
def test_encoder_on_current_position_page():
    """
    Test model on a single given location 
    On broser:
    http://127.0.0.1:5000/test_encoder_on_current_position
    On command line:
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/test_data/locations_example_single.json "http://127.0.0.1:5000/test_encoder_on_current_position"
    
    Copied mostly from test_encoder_on_location
    """

    try:
        data       = request.json
        test_df    = pd.DataFrame(data, index=[0])
        test_name  = "_".join(test_df[["x", "y", "z"]].astype(int).values[0].astype(str))

        test_path = f"./utils/assets/test_data/location_single_{test_name}.csv"

        # Generate points close to the 
        curr_loc           = test_df
        curr_neigborhood   = pd.concat(9*[curr_loc]).reset_index(drop=True) 

        neigborhood_deltas = np.hstack([
                    np.vstack([(i,j) for i in range(-1, 2) for j in range(-1, 2)]),\
                    np.zeros(9).reshape((-1, 1))\
        ])

        curr_neigborhood[["x", "y", "z"]] = curr_neigborhood[["x", "y", "z"]] + neigborhood_deltas

        test_df = curr_neigborhood
        # test_df.to_csv(test_path, index=False)
        #print("\n$$$$$$$$$$$$$$$$$$$$$\n")
            
    except Exception as e:
        print(e)
        print("Empty JSON sent in the request - Using the Example locations file")
        test_name = "example"
        test_path = f'./utils/assets/test_data/locations_example_single.csv'

    start_time = time.time()
    # nmt = request.args.get('nmt', 'percentages')       #normalization type
    # bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    # test_path = f"./utils/assets/test_data/locations_{test_name}.csv"
    # info_dict_path = "./utils/assets/data/semantics_final/models/training_info_1000.json" # December 2024 path

    info_dict_path = "./utils/assets/data/semantics/models/training_info_1000.json"

    trained_encoder, info_dict = load_model_from_info_dict_path(info_dict_path)
    
    test_predictions = []
    locations = test_df[["x", "y", "z", "xh", "yh", "zh"]].values.tolist()
    
    exact_prediction = True
    # exact_prediction = False
    if exact_prediction:
        locations = curr_loc[["x", "y", "z", "xh", "yh", "zh"]].values
        print("predictions are NOT smoothened")
        # xyz  = torch.tensor(curr_loc[["x" , "y", "z"]].values[0])
        # xyzh = torch.tensor(curr_loc[["xh" , "yh", "zh"]].values[0])

        # _,_, prediction = trained_encoder.predict_from_raw(xyz, xyzh)
        
        # test_predictions.append((1+prediction[0].detach().numpy())*.5)
    else: #Smoothened out prediction
        #There are 9 locations of the current neighborhood (see curr_neigborhood above)
        print("predictions are neighborhood smoothened")
    for location in locations:
        
        xyz  = torch.tensor(location[:3])
        xyzh = torch.tensor(location[3:])

        _,_, prediction = trained_encoder.predict_from_raw(xyz, xyzh)
        
        test_predictions.append((1+prediction[0].detach().numpy())*.5)
    
    test_predictions = np.vstack(test_predictions)
    
    
    #print(ml)
    #mp = "./utils/assets/models/"# path to models folder
    #mv = 350                     #model version
    #mean_loss, all_losses, test_predictions, test_df, info_dict = test_encoder_on_data(test_path, mp, mv, True, bs, nmt)

    # print(test_predictions)
    # print()
    # print(test_predictions.sum(axis=1))
    # print(test_predictions.mean(axis=0))
    # print(test_df)
    
    keys   = test_df[["x", "y", "z", "xh", "yh", "zh"]].values.tolist()[:1]
    #values = [test_predictions.mean(axis=0).tolist()]#.tolist()
    values = [dict(zip(info_dict["non_empty_classes_names"], test_predictions.mean(axis=0).tolist()))]


    print("Input:\n\t",curr_loc)
    percentage_predictions = np.round(100*(prediction.detach().numpy()[0] / 2 + 0.5), 2)
    predicition_dictionary = dict(zip(info_dict["non_empty_classes_names"], percentage_predictions))
    print("\nOutput:\n\t",predicition_dictionary)

    semantic_actuals   = [int(s) for s in eval(curr_loc["f_xyz"].values[0])]
    percentage_actuals = [np.round(100 * (p / sum(semantic_actuals)), 2) for p in semantic_actuals]
    actuals_dictionary = dict(zip(info_dict["non_empty_classes_names"], percentage_actuals))
    print("Actual percentages:\n\t", actuals_dictionary, "\n" )

    return jsonify([{"camera_coordinates": k, "predictions" : v} for k, v in zip(keys, values)])




@app.route("/test_encoder_on_data", methods=['POST', "GET"])
@cross_origin()
def test_encoder_on_data_page(test_path=None):
    """
    Test model on locations.csv
    On broser:
    http://127.0.0.1:5000/test_encoder_on_data
    On command line:
    curl -X POST -H "Content-Type: application/json" --data @./utils/assets/test_data/locations_example.json http://127.0.0.1:5000/test_encoder_on_data
    """
    if test_path is None:
        test_name  = request.args.get('bid', 'unnamed_building')  #buiding id
        test_path = f"./utils/assets/test_data/locations_{test_name}.csv"

        try:
            data       = request.json
            test_df    = pd.DataFrame(data).to_csv(test_path, index=False)
            ml         = bool(request.args.get('ml', True))    #missing labels
            #print()
            
        except:
            print("Empty JSON sent in the request - Using the Example locations file")
            test_name = "example"
            test_path = f'./utils/assets/test_data/locations_{test_name}.csv'
            ml        = False #Missing Labels
    else:
        ml = True #Missing Labels

    start_time = time.time()
    nmt = request.args.get('nmt', 'percentages')       #normalization type
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    # test_path = f"./utils/assets/test_data/locations_{test_name}.csv"
    
    #print(ml)
    mp = "./utils/assets/models/"# path to models folder
    mv = 350                     #model version
    mean_loss, all_losses, test_predictions, test_df, info_dict = test_encoder_on_data(test_path, mp, mv, ml, bs, nmt)

    #Performance logs:
    total_time = time.time() - start_time
    print("Buidling dimensions:", test_path.split("/")[-1][:-4]) 
    xM, yM, zM = test_df.max()[["x","y","z"]]
    xm, ym, zm = test_df.min()[["x","y","z"]]
    print(f"\tdx {xM-xm:.1f}; dy {yM-ym:.1f}; dz {zM-zm:.1f}")
    print(f"\tloss: {mean_loss.mean():.4f}, #points: {test_predictions.shape[0]:,}")
    print(f"\tprediction time: total - {total_time:.2f}s; per location - {total_time/test_df.shape[0]:.7f}\n")

    keys   = test_df[["x", "y", "z", "xh", "yh", "zh"]].values.tolist()
    values = test_predictions.tolist()

    return jsonify([{"camera_coordinates": k, "predictions" : v} for k, v in zip(keys, values)])


@app.route("/reset_encoder_weights")
@cross_origin()
def reset_encoder_weights_page():
    '''
    Reset weights of model to the state before building removal.
    '''
    reset_encoder_weights()
    return jsonify({"Restored weights in":"./utils/assets/models/encoder_350.pt"})


if __name__ == '__main__':
    app.run(debug=True)

