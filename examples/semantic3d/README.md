# Semantic3D
Data can be downloaded at [http://semantic3d.net](http://semantic3d.net).
semantic-8 is a benchmark for classification with 8 class labels, namely
{1: man-made terrain, 2: natural terrain, 3: high vegetation, 4: low vegetation, 5: buildings, 6: hard scape, 7: scanning artefacts, 8: cars}.

#Complie some C++ lib
1.To compile Semantic3D_utils

In the folder "semantic3D_utils":

    python setup.py install --home="."

2.To compile the K-nearest neighbors library

In the folder "knn"：

    python setup.py install --home"."

#Data prepare
After downloading the semantic3d data, organize the folder like:
```
dataset    <-- data root dir
├── results    <-- results root dir
│   ├── benchmark_label    <-- dense test labels dir (after interpolation), for submitting to benchmark
│   │   ├── full    <-- for semantic-8
│   │   └── reduced    <-- for reduced-8
│   │       ├── marketsquarefeldkirch4-reduced.labels
│   │       ├── ...
│   │       └── stgallencathedral6-reduced.labels
│   │
│   ├── models    <-- trained model root dir
│   │   ├── ...
│   │   └── PointNet_8192_nocolorFalse_2019-12-16-00-50-51    <-- e.g. model dir
│   │       ├── log.txt    <-- train log file
│   │       └── state_dict.pth    <-- trained model
│   │
│   └── predicted_label    <-- all predicted sparse labels root dir, both semantic-8 and reduced-8
│       ├── MarketplaceFeldkirch_Station4_rgb_intensity-reduced_voxels.npy
│       ├── ...
│       └── StGallenCathedral_station6_rgb_intensity-reduced_voxels.npy
│
├── semantic3d_processed    <-- preprocessed data root dir
│   ├── test    <-- test voxelized data folder both semantic-8 and reduced-8
│   │   ├── pointcloud_npy    <-- voxelized npy data
│   │   │   ├── MarketplaceFeldkirch_Station4_rgb_intensity-reduced_voxels.npy
│   │   │   ├── ...
│   │   │   └── StGallenCathedral_station6_rgb_intensity-reduced_voxels.npy
│   │   └── pointcloud_txt    <-- voxelized txt data
│   │       ├── MarketplaceFeldkirch_Station4_rgb_intensity-reduced_voxels.txt
│   │       ├── ...
│   │       └── StGallenCathedral_station6_rgb_intensity-reduced_voxels.txt
│   │
│   └── train    <-- all train voxelized data folder, contains x,y,z,r,g,b and its labels after voxelization
│       ├── pointcloud_npy
│       │   ├── bildstein_station1_xyz_intensity_rgb_voxels.npy
│       │   ├── ...
│       │   └── untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy
│       └── pointcloud_txt
│           ├── bildstein_station1_xyz_intensity_rgb_voxels.txt
│           ├── ...
│           └── untermaederbrunnen_station3_xyz_intensity_rgb_voxels.txt
│  
└── semantic3d_raw    <-- raw data root dir
    ├── test    <-- test data folder both semantic-8 and reduced-8
    │   ├── birdfountain_station1_xyz_intensity_rgb.txt
    │   ├── ...
    │   └── StGallenCathedral_station6_rgb_intensity-reduced.txt
    │  
    └── train    <-- all raw train data and corresponded labels folder
        ├── bildstein_station1_xyz_intensity_rgb.labels
        ├── bildstein_station1_xyz_intensity_rgb.txt
        ├── ...
        ├── untermaederbrunnen_station3_xyz_intensity_rgb.labels
        └── untermaederbrunnen_station3_xyz_intensity_rgb.txt
```
#Run
Then, run the generation script:

    python semantic3d_prepare_data.py --rootdir raw_data_root_dir --savedir preprocessed_data_root_dir

## Training

The training script is called using:

    python semantic3d_seg.py --rootdir path_to_preprocessed_data_root_dir --savedir path_to_results_root_dir --nocolor --isreduced

(nocolor means color feature not used, isreduced means using reduced-8 test data)



## Test

To predict on the test set (voxelized pointcloud):


    python semantic3d_seg.py --rootdir path_to_preprocessed_data_root_dir --savedir path_to_results_root_dir --test


Finally to generate the prediction files at benchmark format (may take som time): 


    python semantic3d_benchmark_gen.py --dense_data_dir path_to_raw_test_data --dense_label_dir path_to_save_benchmark_label --sparse_data_dir preprocessed_test_txt_data_dir --sparse_label_dir path_to_prediction_label


## Acknowledgement
* [aboulch/ConvPoint](https://github.com/aboulch/ConvPoint)
