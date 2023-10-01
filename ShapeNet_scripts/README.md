# ShapeNet_scripts

This folder contains scripts to generate point clouds for simulated imaging.

## Installion

Follow ["Sampling color and geometry point clouds from ShapeNet dataset"](https://github.com/mmspg/mesh-sampling) to install the environment with pymeshlab 2021.10 for get_mesh_sampling.py.  
Install another environment with pymeshlab 2022.2.post4 for get_ply_from_mesh.py.

## Usage

1. Download [ShapeNetCore.v2](https://shapenet.org/).
2. Change models_path in get_mesh_sampling.py and get_ply_from_mesh.py to your ShapeNet path.
3. First run "python get_mesh_sampling.py".
4. Second run "python get_ply_from_mesh.py".
