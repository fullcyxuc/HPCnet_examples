import numpy as np
import argparse
import os
import semantic3D_utils.lib.python.semantic3D as Sem3D

'''
preprocess the raw data(voxelization)
'''

parser = argparse.ArgumentParser()
# raw data dir
parser.add_argument('--rootdir', type=str, default="./dataset/semantic3d_raw", help='Path to data folder')

# preprocessed data dir
parser.add_argument("--savedir", type=str, default="./dataset/semantic3d_processed", help="Path to saved data folder")

# size for voxelization
parser.add_argument("--voxel", type=float, default=0.03, help="voxel size")

# whether use the reduced data
parser.add_argument("--isreduced", action='store_true', help="set for using the reduced test data")
args = parser.parse_args()

filelist_train=[
        "bildstein_station1_xyz_intensity_rgb",
        "bildstein_station3_xyz_intensity_rgb",
        "bildstein_station5_xyz_intensity_rgb",
        "domfountain_station1_xyz_intensity_rgb",
        "domfountain_station2_xyz_intensity_rgb",
        "domfountain_station3_xyz_intensity_rgb",
        "neugasse_station1_xyz_intensity_rgb",
        "sg27_station1_intensity_rgb",
        "sg27_station2_intensity_rgb",
        "sg27_station4_intensity_rgb",
        "sg27_station5_intensity_rgb",
        "sg27_station9_intensity_rgb",
        "sg28_station4_intensity_rgb",
        "untermaederbrunnen_station1_xyz_intensity_rgb",
        "untermaederbrunnen_station3_xyz_intensity_rgb",
    ]

filelist_test = [
        "birdfountain_station1_xyz_intensity_rgb",
        "castleblatten_station1_intensity_rgb",
        "castleblatten_station5_xyz_intensity_rgb",
        "marketplacefeldkirch_station1_intensity_rgb",
        "marketplacefeldkirch_station4_intensity_rgb",
        "marketplacefeldkirch_station7_intensity_rgb",
        "sg27_station10_intensity_rgb",
        "sg27_station3_intensity_rgb",
        "sg27_station6_intensity_rgb",
        "sg27_station8_intensity_rgb",
        "sg28_station2_intensity_rgb",
        "sg28_station5_xyz_intensity_rgb",
        "stgallencathedral_station1_intensity_rgb",
        "stgallencathedral_station3_intensity_rgb",
        "stgallencathedral_station6_intensity_rgb",
        ]

filelist_test_reduced = [
    "MarketplaceFeldkirch_Station4_rgb_intensity-reduced",
    "sg27_station10_rgb_intensity-reduced",
    "sg28_Station2_rgb_intensity-reduced",
    "StGallenCathedral_station6_rgb_intensity-reduced",
]

print("Generating train files...")
for filename in filelist_train:
    print(filename)

    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    # load file and voxelize
    savedir = os.path.join(args.savedir, "train", "pointcloud_txt")
    os.makedirs(savedir, exist_ok=True)

    Sem3D.semantic3d_load_from_txt_voxel_labels(os.path.join(args.rootdir, "train", filename_txt),
                                                os.path.join(args.rootdir, "train", filename_labels),
                                                os.path.join(savedir, filename+"_voxels.txt"),
                                                args.voxel
                                                )

    # save the numpy data
    savedir_numpy = os.path.join(args.savedir, "train", "pointcloud_npy")
    os.makedirs(savedir_numpy, exist_ok=True)
    np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))

print("Done")

print("Generating test files...")
# choose the test data(full or reduced)
if args.isreduced:
    filelist_test_chose = filelist_test_reduced
else:
    filelist_test_chose = filelist_test

for filename in filelist_test_chose:
    print(filename)
    
    filename_txt = filename+".txt"

    # load file and voxelize
    savedir = os.path.join(args.savedir, "test", "pointcloud_txt")
    os.makedirs(savedir, exist_ok=True)

    Sem3D.semantic3d_load_from_txt_voxel(os.path.join(args.rootdir, "test", filename_txt),
                                        os.path.join(savedir, filename+"_voxels.txt"),
                                        args.voxel
                                        )
    
    # save the numpy data
    savedir_numpy = os.path.join(args.savedir, "test", "pointcloud_npy")
    os.makedirs(savedir_numpy, exist_ok=True)
    np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))

print("Done")


