# Visibility Encoder
Train and predict visibility for camera parameters within urban contexts.

# Usage 

1. Clone reposityory, create and activate __conda environment__:

```
git clone git@github.com:StefanCobeli/Visibility_Encoder.git && cd Visibility_Encoder
conda env create -f environment.yml
conda activate visibility_encoder
```

2. Run `server.py`

```
python server.py
```

### 3. Use cases:

1. __Facade Generation__: predict visibility values for the facade of a buildilg given by 4 base points.

Example:


`http://127.0.0.1:5000/predict_facade_from_base_points/base_points_example`

`base_point.csv` files need to be located in `/utils/assets/new_buildings/`

2. __Test Encoder__: test and predict visibility values from a locations csv file.

Example:

`http://127.0.0.1:5000/test_encoder_on_data/locations_example`

`location.csv` files need to be located in `/utils/assets/test_data/`


# Output format

All the methods use the following output format, namely a list of the genereated locations. 

__Each location__ is represented by:

1. `camera_coordinates` - The ordere is x, y, z, xh, yh, zh - __xyz__ are the coordinates and __xhyhzh__ are the roatations of the camera.

2. `predictions` - Values between 0 and 1 correspondet to the visibility values towards the selected classes _['building' ' water' ' tree' ' sky']_. 

```
[
  {
    "camera_coordinates": [
      842.5106601410772,
      45.02010124456787,
      1458.2977995297426,
      32.2071524443495,
      83.45431161148385,
      -32.04065846001832
    ],
    "predictions": [
      0.987492561340332,
      0.1578688621520996,
      0.8025062680244446,
      0.6580542922019958
    ]
  },
  {
    "camera_coordinates": [
...
]
```

# Intuitions

New building from base points:

![New building from base points](utils/assets/images/building_from_base_points.png)