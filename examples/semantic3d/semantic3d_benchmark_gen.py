import numpy as np
import semantic3D_utils.lib.python.semantic3D as sem3D
import os
import argparse
import multiprocessing
import time

# multiprocess
theads_num = 4

parser = argparse.ArgumentParser()
# path_to_original_test_data_dir
parser.add_argument('--dense_data_dir', '-s', default="./dataset/semantic3d_raw/test")
# /path_to_save_benchmark_dir
parser.add_argument("--dense_label_dir", type=str, default="./dataset/results/benchmark_label")
# path_to_data_processed_dir
parser.add_argument("--sparse_data_dir", type=str, default="./dataset/semantic3d_processed/test/pointcloud_txt")
# path_to_label_prediction_dir
parser.add_argument("--sparse_label_dir", type=str, default="./dataset/results/predicted_label")
parser.add_argument("--isreduced", action='store_true', help="whether use the reduced test data")
args = parser.parse_args()

filenames = [
        ["birdfountain_station1_xyz_intensity_rgb", "birdfountain1.labels"],
        ["castleblatten_station1_intensity_rgb", "castleblatten1.labels"],
        ["castleblatten_station5_xyz_intensity_rgb", "castleblatten5.labels"],
        ["marketplacefeldkirch_station1_intensity_rgb", "marketsquarefeldkirch1.labels"],
        ["marketplacefeldkirch_station4_intensity_rgb", "marketsquarefeldkirch4.labels"],
        ["marketplacefeldkirch_station7_intensity_rgb", "marketsquarefeldkirch7.labels"],
        ["sg27_station10_intensity_rgb", "sg27_10.labels"],
        ["sg27_station3_intensity_rgb", "sg27_3.labels"],
        ["sg27_station6_intensity_rgb", "sg27_6.labels"],
        ["sg27_station8_intensity_rgb", "sg27_8.labels"],
        ["sg28_station2_intensity_rgb", "sg28_2.labels"],
        ["sg28_station5_xyz_intensity_rgb","sg28_5.labels"],
        ["stgallencathedral_station1_intensity_rgb", "stgallencathedral1.labels"],
        ["stgallencathedral_station3_intensity_rgb", "stgallencathedral3.labels"],
        ["stgallencathedral_station6_intensity_rgb", "stgallencathedral6.labels"],
]

filenames_reduced = [
    ["MarketplaceFeldkirch_Station4_rgb_intensity-reduced", "marketsquarefeldkirch4-reduced.labels"],
    ["sg27_station10_rgb_intensity-reduced", "sg27_10-reduced.labels"],
    ["sg28_Station2_rgb_intensity-reduced", "sg28_2-reduced.labels"],
    ["StGallenCathedral_station6_rgb_intensity-reduced", "stgallencathedral6-reduced.labels"],
]


os.makedirs(args.dense_label_dir, exist_ok=True)

if args.isreduced:
    filelist_chose = filenames_reduced
else:
    filelist_chose = filenames

t1 = time.clock()

for fname in filelist_chose:
    print(fname[0])
    data_filename = os.path.join(args.dense_data_dir, fname[0]+".txt")  # raw test file
    if args.isreduced:
        dest_filaname = os.path.join(args.dense_label_dir, "reduced", fname[1])  # dense predicted labels to saved
    else:
        dest_filaname = os.path.join(args.dense_label_dir, "full", fname[1])  # dense predicted labels to saved

    refdata_filename = os.path.join(args.sparse_data_dir, fname[0]+"_voxels.txt")  # test data preprocessed
    reflabel_filename = os.path.join(args.sparse_label_dir, fname[0]+"_voxels.npy")  # sparse predicted labels

    sem3D.project_labels_to_pc(dest_filaname, data_filename, refdata_filename, reflabel_filename)

print("consume time:", time.clock() - t1, "s")