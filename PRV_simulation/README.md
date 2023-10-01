# PRV_simulation

This folder contains the simulation system for our PRV-based active NeRF reconstruction.

## Installion

These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 10.0.0 (free for academic use), and JsonCpp.

Our codes can be compiled by Visual Studio 2022 with c++ 14 and run on Windows 11.

For other system, please check the file read/write or multithreading functions in the codes.

## Main Usage

DefaultConfiguration.yaml contains most parameters.

The mode of the system should be input in the Console. These modes are for different functions as follows.  

Then give the object model names in the Console (-1 to break input).

## Dataset Generation

You may download our processed Required Number of Views Dataset from [Kaggle](). Or use the following instructions to process 3D models.

### A. Object 3D Model Dataset

1. Follow [ShapeNet_scripts](https://github.com/psc0628/NeRF-PRV/tree/main/ShapeNet_scripts) folder to obtain processed point clouds for ShapeNet models.
2. Change shape_net path in DefaultConfiguration.yaml.
3. Run with mode = 10 (ShapeNetPreProcess).
4. Input [04379243 02958343 03001627 02691156 04256520 04090263 03636649 04530566 02828884 03691459 02933112 03211117 04401088 02924116 02808440 03467517 03325088 03046257 03991062 03593526] and -1.

### B. Required Number of Views Dataset

Note that this process takes a lot of time for a large number of objects.

#### B.1 View Space

Use our processed view spaces in Hemisphere subfolder.  
Or you want to generate them:

1. Download Tammes_sphere from [Tammes-problem](https://github.com/XiangjingLai/Tammes-problem).
2. Change orginalviews_path in DefaultConfiguration.yaml.
3. Run with mode = 0 (ViewCover) with -1.

#### B.2 Size Augmentation

1. Frist Run with mode = 2 (GetSizeTest).
2. Input names in A generated ShapeNet_names.txt and -1.
3. Second Run with mode = 11 (GetCleanData).
4. Input names in A generated ShapeNet_names.txt and -1.

#### B.3 Generate Image for Each View

1. Run with mode = 3 (GetCoverage).
2. Input names in B.2 generated clean_names.txt and -1.

#### B.4 Plot PSNR for Each Object

Follow [instantngp_scripts](https://github.com/psc0628/NeRF-PRV/tree/main/instantngp_scripts) folder to setup instantngp environment.

1. Change instant_ngp_path in DefaultConfiguration.yaml.
2. Run with mode = 4 (InstantNGP).
3. Input names in B.2 generated clean_names.txt and -1.
4. Run "python train_server.py" at the same time.
