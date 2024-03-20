
from flask import Flask, request, jsonify
from utils.scripts.architectures.train_location_encoder import *
from utils.test_location_encoder                        import *


app = Flask(__name__)


# Create locations on facade of the building 
@app.route('/predict_facade_from_base_points', methods=['POST', "GET"])
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
        bp_df = pd.DataFrame(data)#.to_dict(orient="records")
    except:
        print("Empty JSON sent in the request - Using the Example base points")
        base_points_name = "base_points_example"
        base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')
        bp_df            = pd.read_csv(base_points_path, header=None)

    start_time = time.time()
    #1. Read base points
    
    bp  = bp_df.values
    bh  = int(request.args.get('bh', '50'))   #buiding height
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    ppf = int(request.args.get('ppf', '250')) #points per facade side
    nmt = request.args.get('nmt', 'percentages') #normalization type

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
    return jsonify(facade_dict)



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
        test_df    = pd.DataFrame(data)
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
        test_df.to_csv(test_path, index=False)
        #print("\n$$$$$$$$$$$$$$$$$$$$$\n")
            
    except:
        print("Empty JSON sent in the request - Using the Example locations file")
        test_name = "example"
        test_path = f'./utils/assets/test_data/locations_example_single.csv'

    start_time = time.time()
    nmt = request.args.get('nmt', 'percentages')       #normalization type
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    # test_path = f"./utils/assets/test_data/locations_{test_name}.csv"
    
    #print(ml)
    mp = "./utils/assets/models/"# path to models folder
    mv = 350                     #model version
    mean_loss, all_losses, test_predictions, test_df, info_dict = test_encoder_on_data(test_path, mp, mv, True, bs, nmt)

    # print(test_predictions)
    # print()
    # print(test_predictions.sum(axis=1))
    # print(test_predictions.mean(axis=0))
    # print(test_df)
    
    keys   = test_df[["x", "y", "z", "xh", "yh", "zh"]].values.tolist()[:1]
    values = [test_predictions.mean(axis=0).tolist()]#.tolist()

    return jsonify([{"camera_coordinates": k, "predictions" : v} for k, v in zip(keys, values)])



@app.route("/test_encoder_on_data", methods=['POST', "GET"])
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


if __name__ == '__main__':
    app.run(debug=True)

