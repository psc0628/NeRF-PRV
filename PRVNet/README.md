# PRVNet

This folder contains our PRVNet and interfaces.

## Installion

Follow ["ConvNeXt-V2"](https://github.com/facebookresearch/ConvNeXt-V2) to install the environment and move all files into the ConvNeXt-V2 folder.  

## Training

1. Download ["ConvNeXt V2-T ImageNet-1K fine-tuned model"](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt) and put into ImageNet folder.
2. Download [Required Number of Views Dataset]() or perpare via [PRV_simulation](https://github.com/psc0628/NeRF-PRV/tree/main/PRV_simulation) and put into data_5view.
3. Run "python train_regression.py --model convnextv2_tiny --data_path ./data_5view --premodel_file ./ImageNet/convnextv2_tiny_1k_224_ema.pt --output_dir ./output/ImageNet_fine_l1 --log_dir ./log/ImageNet_fine_l1 --loss_type L1 --ImageNet"

## Our Trained PRVNet for View Planning Interface

Download [best_checkpoint.pth](https://drive.google.com/file/d/1VTYlfycuB3xitMubeY6xx7Ku-qAprnR9/view?usp=drive_link) and put into checkpoints folder.
