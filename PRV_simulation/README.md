# PRV_simulation

This folder contains the simulation system for our PRV-based active NeRF reconstruction.

## Installion

These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 10.0.0 (free for academic use), and JsonCpp.
Our codes can be compiled by Visual Studio 2022 with c++ 14 and run on Windows 11.
For other system, please check the file read/write or multithreading functions in the codes.

## Main Usage

DefaultConfiguration.yaml contains most parameters.  

The mode of the system should be input in the Console. These modes are for different functions as follows.
## Dataset Generation

You may download our processed Required Number of Views Dataset from [Kaggle](). Or use the following instructions to process 3D models.

### Object 3D Model Dataset

Follow ShapeNet_scripts folder to obtain processed point clouds for ShapeNet models.
Change shape_net path in DefaultConfiguration.yaml.
Run with mode = 10 (ShapeNetPreProcess).

### 


