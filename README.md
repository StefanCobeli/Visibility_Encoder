# A Neural Field-Based Approach for View Computation & Data Exploration in 3D Urban Environments

This repository contains the implementation of the approaches introduced in the work **A Neural Field-Based Approach for
View Computation & Data Exploration in 3D Urban Environments** by Stefan Cobeli, Kazi Shahrukh Omar, Rodrigo Valença, Nivan Ferreira and Fabio Miranda.  

The main method encodes visibility distributions in 3d environments(e.g., towards buildings, trees, water, or sky) and supports direct queries (computing visibility from a given viewpoint) as well as inverse queries (finding viewpoints that match desired conditions).  

![Figure 1](./figures/use-case-1_compressed.png)  
*Example of direct queries for computing thematic visibility data associated with building facades.*

---

## Installation & Setup

We recommend using **Anaconda** with Python `3.9.13`.

1. Clone the repository:

   ```bash
   git clone https://github.com/StefanCobeli/Visibility_Encoder.git
   git clone https://github.com/StefanCobeli/visibility-data-generator.git
   cd Visibility_Encoder

2.	Create the environment:

    ```bash
    conda create -n Visibility_Encoder python=3.9.13

3.	Run the setup script:

    ```bash
    ./run.sh

## Usage

After running `run.sh`, you can access the interface and compute visibility distributions at 
      
      http://localhost:5173



## Data & Meshes:

The visibility dataset can be found at:

    utils/assets/data/locations.csv

The 3D models can be found at:

     ../visibility-data-generator/static/data/NYC/collada_model/      
