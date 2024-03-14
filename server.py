
from flask import Flask, request, jsonify
from utils.scripts.architectures.train_location_encoder import *
from utils.test_location_encoder                        import *


app = Flask(__name__)

@app.route('/test_base_json', methods=['POST'])
def test_base_json():
    """
    https://sentry.io/answers/flask-getting-post-data/
    """
    try:
        data = request.json
    except:
        print("Empty JSON")
        return "Empty"

    print(data[0])
    print(data[1])
    # print(data.get('age'))
    df = pd.DataFrame(data)
    print(df)
    return jsonify(df.to_dict(orient="records"))
    # return jsonify(data)


@app.route('/test_json', methods=['POST'])
def test_json():
    """
    https://sentry.io/answers/flask-getting-post-data/
    """
    data = request.json
    print(data.get('name'))
    print(data.get('age'))
    return data


# Create locations on facade of the building 
@app.route('/predict_facade_from_base_points', methods=['POST', "GET"])
def predict_facade_from_base_points_page():
    '''
    generated predictions and inputs on facade from base points 
    http://127.0.0.1:5000/predict_facade_from_base_points/base_points_example?bh=50&ppf=250
    '''

    base_points_name = request.args.get('bpn', 'unnamed_points') #"base_points_example"
    base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')

    try:
        data  = request.json
        bp_df = pd.DataFrame(data)#.to_dict(orient="records")
    except:
        print("Empty JSON sent in the request - Using the Example base points")
        #base_points_name = "base_points_example"
        base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')
        bp_df            = pd.read_csv(base_points_path, header=None)

    start_time = time.time()
    #1. Read base points
    
    bp  = bp_df.values
    bh  = int(request.args.get('bh', '50'))   #buiding height
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    ppf = int(request.args.get('ppf', '250')) #points per facade side
    nmt = request.args.get('nmt', 'log') #normalization type

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

@app.route("/test_encoder_on_data", methods=['POST', "GET"])
def test_encoder_on_data_page():
    """
    Test model on locations.csv
    http://127.0.0.1:5000/test_encoder_on_data
    """

    test_name  = request.args.get('bid', 'unnamed_building')  #buiding id
    test_path = f"./utils/assets/test_data/locations_{test_name}.csv"

    try:
        data       = request.json
        test_df    = pd.DataFrame(data).to_csv(test_path, index=False)
    except:
        print("Empty JSON sent in the request - Using the Example locations file")
        test_name = "example"
        test_path = f'./utils/assets/test_data/locations_{test_name}.csv'
        # bp_df     = pd.read_csv(base_points_path, header=None) 

    start_time = time.time()
    nmt = request.args.get('nmt', 'log')       #normalization type
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    # test_path = f"./utils/assets/test_data/locations_{test_name}.csv"
    
    mp = "./utils/assets/models/"# path to models folder
    mv = 350                     #model version
    mean_loss, all_losses, test_predictions, test_df, info_dict = test_encoder_on_data(test_path, mp, mv, False, bs, nmt)

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

