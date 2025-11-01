# DGMLP-SSTN: Enhanced 3D Human Pose Estimation via Dynamic Graph MLP with Spatial Self-Transform



<p align="center"><img src="figure/model.png" width="100%" alt="" /></p>

## Installation

DGMLP-SSTN is tested on Ubuntu 18 with Pytorch 1.7.1 and Python 3.9. 
- Create a conda environment: ```conda create -n DGMLP-SSTN python=3.9```
- Install PyTorch 2.2.2 and Torchvision 0.17.2 following the [official instructions](https://pytorch.org/)
- ```pip3 install -r requirements.txt```
  
## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) website, and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/18mvXIZ98LKGAqDFpRsNVvCRonVBAlgoX?usp=share_link). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_3d_3dhp.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
|   |-- data_2d_3dhp.npz
```

## Train the model

To train a 1-frame DGMLP-SSTN model on Human3.6M:

```bash
# Train from scratch
python main.py --batch_size 256

# After training for 20 epochs, add refine module
python main.py  --batch_size 256 --refine --lr 1e-5 --previous_dir [your best model saved path]
```

## Test the model

To test a 1-frame DGMLP-SSTN model:

```bash
# Human3.6M
python main.py --test --previous_dir '[your best model saved path]' 

# MPI-INF-3DHP
python main.py --test --previous_dir '[your best model saved path]'  --dataset '3dhp'
```

To test a 1-frame GraphMLP model with refine module on Human3.6M:
```bash
python main.py --test --previous_dir '[your best model saved path]' --refine 
```



## Demo
First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. 
Then, you need to put your in-the-wild videos in the './demo/video' directory. 

Run the command below:
```bash
# Run the command below:
python demo/vis.py --video sample_video.mp4

# Or run the command with the fixed z-axis:
python demo/vis.py --video sample_video.mp4 --fix_z
```


<!-- <p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>

Or run the command below:
```bash
python demo/vis.py --video sample_video.mp4 --fix_z
```

Sample demo output:

<p align="center"><img src="figure/sample_video_fix_z.gif" width="60%" alt="" /></p> -->


## Citation

If you find our work useful in your research, please consider citing:

    @article{zhao2025dgmlpsstn},
      title={DGMLP-SSTN: Enhanced 3D Human Pose Estimation via Dynamic Graph MLP with Spatial Self-Transform},
      author={Zhang, Xiaoli and Zhao, Siya and Zhu, Guifu and Wang, Sicui and Yang, Yunfei},
      journal={},
      pages={},
      year={2025}
    }

## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [GraphMLP](https://github.com/Vegetebird/GraphMLP)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
## Licence

This project is licensed under the terms of the MIT license.
