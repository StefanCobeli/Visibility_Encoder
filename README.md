# Visibility_Encoder
Train and predict visibility for camera parameters within urban contexts.

# Usage 

1. Clone reposityory, create and activate conda environment:

```
git clone git@github.com:StefanCobeli/Visibility_Encoder.git && cd Visibility_Encoder
conda env create -f environment.yml
conda activate visibility_encoder
```

2. Run `server.py`

```
python server.py
```

3. Use cases:

(i). Predict visibility values for the facade of a buildilg given by 4 base points:

Example:
`http://127.0.0.1:5000/predict_facade_from_base_points/base_points_example`

`base_point.csv` files need to be located in `/utils/assets/new_buildings/`

(ii). Test and predict visibility values from a locations csv file:

Example:
`http://127.0.0.1:5000/test_encoder_on_data/locations_example`

`location.csv` files need to be located in `/utils/assets/test_data/`
