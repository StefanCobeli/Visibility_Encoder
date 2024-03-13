
from flask import Flask, request, jsonify
from utils.scripts.architectures.train_location_encoder import *
from utils.test_location_encoder                        import *


app = Flask(__name__)

# Create locations of 
@app.route('/predict_facade_from_base_points/<base_points_name>')
def predict_facade_from_base_points_page(base_points_name):
    '''
    generated predictions and inputs on facade from base points 
    http://127.0.0.1:5000/predict_facade_from_base_points/base_points_example?bh=50&ppf=250
    '''
    start_time = time.time()
    #1. Read base points
    base_points_path = request.args.get('bph', f'./utils/assets/new_buildings/{base_points_name}.csv')
    bh  = int(request.args.get('bh', '50'))   #buiding height
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    ppf = int(request.args.get('ppf', '250')) #points per facade side
    nmt = request.args.get('nmt', 'log') #normalization type
    bp  = pd.read_csv(base_points_path, header=None).values

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

@app.route("/test_encoder_on_data/<test_name>")
def test_encoder_on_data_page(test_name):
    """
    Test model on locations.csv
    http://127.0.0.1:5000/test_encoder_on_data/locations_example
    """
    start_time = time.time()
    nmt = request.args.get('nmt', 'log')       #normalization type
    bs  = int(request.args.get('bs', '32768')) #batch size - default 2**15 - 32768
    test_path = f"./utils/assets/test_data/{test_name}.csv"
    
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

