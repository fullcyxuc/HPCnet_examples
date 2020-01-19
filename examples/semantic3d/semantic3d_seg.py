# Semantic3D Example with ConvPoint

# add the parent folder to the python path to access pointnet library
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

import numpy as np
import argparse
from datetime import datetime
import random
from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms

from sklearn.metrics import confusion_matrix
import utils.metrics as metrics
import knn.lib.python.nearest_neighbors as nearest_neighbors
from HPCnet.models.hpc_pointnet2_msg_sem import get_model

from PIL import Image

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

device_ids = [0]

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    ''' interpolate the test score from the nearest selected points for all points not selected
        pts_src: points selected for testing
        pts_dest: all points in a test file
        data_src: scores of points selected
    '''
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    print(indices.shape)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[ cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [      0,      0, 1],])
    return np.dot(batch_data, rotation_matrix)

# Part dataset only for training / validation
class PartDataset():

    def __init__ (self, filelist, folder,
                    training=False, 
                    iteration_number = None,
                    block_size=8,
                    npoints = 8192,
                    nocolor=False):

        self.folder = folder
        self.training = training
        self.filelist = filelist
        self.bs = block_size
        self.nocolor = nocolor

        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False

        self.transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4)

    def __getitem__(self, index):
        
        # load the data
        index = random.randint(0, len(self.filelist)-1)
        pts = np.load(os.path.join(self.folder, self.filelist[index]))

        # get the features: color rgb
        rgb = pts[:,3:6]

        # get the labels
        lbs = pts[:, 6].astype(int)-1 # the generation script label starts at 1

        # get the point coordinates
        pts = pts[:, :3]

        # pick a random point
        pt_id = random.randint(0, pts.shape[0]-1)
        pt = pts[pt_id]

        # create the mask 只取以该随机为中点的一个block范围内竖直方向所有点
        '''
        as the paper says:
            "At training time, we randomly select points in the considered
            point cloud, and extract all the points in an infinite vertical
            column centered on this point, the column section is 2 meters
            for indoor scenes and 8 meters for outdoor scenes"
        '''
        mask_x = np.logical_and(pts[:,0]<pt[0]+self.bs/2, pts[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(pts[:,1]<pt[1]+self.bs/2, pts[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = pts[mask]
        lbs = lbs[mask]
        rgb = rgb[mask]
        
        # random selection 随机取点用来训练，论文里说是8192个
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]
        rgb = rgb[choice]

        # data augmentation
        if self.training:
            # random rotation
            pts = rotate_point_cloud_z(pts)

            # random jittering
            rgb = rgb.astype(np.uint8)
            rgb = np.array(self.transform( Image.fromarray(np.expand_dims(rgb, 0)) ))
            rgb = np.squeeze(rgb, 0)
        
        rgb = rgb.astype(np.float32)
        rgb = rgb / 255 - 0.5

        if self.nocolor:
            rgb = np.ones((pts.shape[0], 1))

        pts = torch.from_numpy(pts).float()
        rgb = torch.from_numpy(rgb).float()
        lbs = torch.from_numpy(lbs).long()


        return pts, rgb, lbs

    def __len__(self):
        return self.iterations


class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, filename, folder,
                    block_size=8,
                    npoints = 8192,
                    test_step=0.1,
                    nocolor=False):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.nocolor = nocolor
        self.filename = filename

        # load the points 代码的test_step使用0.8而论文是0.5
        '''
            as the paper says:
                we compute a 2D occupancy pixel map with
                pixel size 0.1 meters for indoor scenes and 0.5 meters for outdoor
                scenes by projecting vertically on the horizontal plane. Then,
                we considered each occupied cell as a center for a column (same
                size as for training).
        '''
        self.xyzrgb = np.load(os.path.join(self.folder, self.filename))
        step = test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)  # 只取1个点代表每个二维pixel map
        self.pts = self.pts.astype(np.float)*step

    def __getitem__(self, index):
        
        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        # separate between features and points
        if self.nocolor:
            rgb = np.ones((pts.shape[0], 1))
        else:
            rgb = pts[:, 3:6]
            rgb = rgb.astype(np.float32)
            rgb = rgb / 255 - 0.5

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        rgb = torch.from_numpy(rgb).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, rgb, lbs

    def __len__(self):
        return len(self.pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, default="./dataset/semantic3d_processed", help='Path to preprocessed data folder')
    parser.add_argument("--savedir", type=str, default="./dataset/results", help='Path to saved result folder')
    parser.add_argument('--block_size', help='Block size', type=float, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--iter", "-i", type=int, default=1000)
    parser.add_argument("--npoints", "-n", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--test_step", default=0.8, type=float)
    parser.add_argument("--model", default="PointNet++", type=str, help="model name")
    parser.add_argument("--isreduced", action='store_true', help="whether use the reduced test data")
    parser.add_argument('--modeldir', type=str,
                        default="./dataset/results/models/PointNet_1000_nocolorTrue_2019-12-22-17-39-48",
                        help="dir of models for testing")

    args = parser.parse_args()
    # print(args)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_folder = os.path.join(args.savedir, "models", "{}_{}_nocolor{}_{}".format(
            args.model, args.npoints, args.nocolor, time_string))

    filelist_train=[
        "bildstein_station1_xyz_intensity_rgb_voxels.npy",
        "bildstein_station3_xyz_intensity_rgb_voxels.npy",
        "bildstein_station5_xyz_intensity_rgb_voxels.npy",
        "domfountain_station1_xyz_intensity_rgb_voxels.npy",
        "domfountain_station2_xyz_intensity_rgb_voxels.npy",
        "domfountain_station3_xyz_intensity_rgb_voxels.npy",
        "neugasse_station1_xyz_intensity_rgb_voxels.npy",
        "sg27_station1_intensity_rgb_voxels.npy",
        "sg27_station2_intensity_rgb_voxels.npy",
        "sg27_station4_intensity_rgb_voxels.npy",
        "sg27_station5_intensity_rgb_voxels.npy",
        "sg27_station9_intensity_rgb_voxels.npy",
        "sg28_station4_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
        "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
    ]

    filelist_test = [
        "birdfountain_station1_xyz_intensity_rgb_voxels.npy",
        "castleblatten_station1_intensity_rgb_voxels.npy",
        "castleblatten_station5_xyz_intensity_rgb_voxels.npy",
        "marketplacefeldkirch_station1_intensity_rgb_voxels.npy",
        "marketplacefeldkirch_station4_intensity_rgb_voxels.npy",
        "marketplacefeldkirch_station7_intensity_rgb_voxels.npy",
        "sg27_station10_intensity_rgb_voxels.npy",
        "sg27_station3_intensity_rgb_voxels.npy",
        "sg27_station6_intensity_rgb_voxels.npy",
        "sg27_station8_intensity_rgb_voxels.npy",
        "sg28_station2_intensity_rgb_voxels.npy",
        "sg28_station5_xyz_intensity_rgb_voxels.npy",
        "stgallencathedral_station1_intensity_rgb_voxels.npy",
        "stgallencathedral_station3_intensity_rgb_voxels.npy",
        "stgallencathedral_station6_intensity_rgb_voxels.npy",
        ]

    filelist_test_reduced = [
        "MarketplaceFeldkirch_Station4_rgb_intensity-reduced_voxels.npy",
        "sg27_station10_rgb_intensity-reduced_voxels.npy",
        "sg28_Station2_rgb_intensity-reduced_voxels.npy",
        "StGallenCathedral_station6_rgb_intensity-reduced_voxels.npy",
    ]

    N_CLASSES = 8

    # create model
    print("Creating the network...", end="", flush=True)
    if args.nocolor:
        net = get_model(num_classes=N_CLASSES)
    else:
        net = get_model(num_classes=N_CLASSES)  # TODO add the color feature

    net = torch.nn.DataParallel(net, device_ids=device_ids)  # 声明所有可用设备
    net.cuda()

    if args.test:
        net.load_state_dict(torch.load(os.path.join(args.modeldir, "state_dict.pth")))  # test with the trained model

    print("Done")

    ##### TRAIN
    if not args.test:

        print("Create the datasets...", end="", flush=True)
        ds = PartDataset(filelist_train, os.path.join(args.rootdir, "train", "pointcloud_npy"),
                                training=True, block_size=args.block_size,
                                iteration_number=args.batch_size*args.iter,
                                npoints=args.npoints,
                                nocolor=args.nocolor)

        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.threads
                                                   )
        print("Done")

        print("Create optimizer...", end="", flush=True)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        print("Done")
        
        # create the model folder
        os.makedirs(model_folder, exist_ok=True)
        
        # create the log file
        logs = open(os.path.join(model_folder, "log.txt"), "w")

        # iterate over epochs
        for epoch in range(args.epochs):
            net.train()
            train_loss = 0
            cm = np.zeros((N_CLASSES, N_CLASSES))

            t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
            oa = None
            aa = None
            iou = None
            d_loss = None

            for pts, rgb, lbs in t:
                # pts = pts.transpose(2, 1)  # torch.Size([16, 3, 8192])
                # rgb = rgb.transpose(2, 1)  # torch.Size([16, 3, 8192])
                # gtFeature = getGtFeature(pts)  # torch.Size([16, dim_num, 8192])
                #
                # if not args.nocolor:
                #     feature = torch.cat((gtFeature, rgb), 1)  # 将颜色拼接
                # else:
                #     feature = gtFeature
                feature = pts  # torch.size: [batch, npoint, dim]

                feature = feature.cuda()
                lbs = lbs.cuda()

                optimizer.zero_grad()
                outputs = net(feature)
                # loss = F.nll_loss(outputs.view(-1, N_CLASSES), lbs.view(-1))
                loss = F.cross_entropy(outputs.view(-1, N_CLASSES), lbs.view(-1))

                loss.backward()
                optimizer.step()
                # 预测label
                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = lbs.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                cm += cm_

                '''
                总体精度（OA）与平均精度（AA）不同
                总体精度是模型在所有测试集上预测正确的与总体数量之间的比值
                平均精度是每一类预测正确的与每一类总体数量之间的比值，最终再取每一类的精度的平均值
                '''
                # oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
                # aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
                # iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"
                oa = "%.5f" % (metrics.stats_overall_accuracy(cm))
                aa = "%.5f" % (metrics.stats_accuracy_per_class(cm)[0])
                iou = "%.5f" % (metrics.stats_iou_per_class(cm)[0])

                train_loss += loss.detach().cpu().item()
                d_loss = "%.4e" % (train_loss/cm.sum())

                # t.set_postfix(AA=wblue(aa), IOU=wblue(iou), OA=wblue(oa), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))
                t.set_postfix(AA=wblue(aa), IOU=wblue(iou), OA=wblue(oa), LOSS=wblue(d_loss))
            # save the model
            torch.save(net.state_dict(), os.path.join(model_folder, "state_dict.pth"))

            # write the logs
            logs.write(f"{epoch} {aa} {iou} {oa} {train_loss/cm.sum():.4e}\n")
            #logs.write(str(epoch) + " " + str(aa) + " " + str(iou) + " " + str(oa) + " " + str(d_loss))
            logs.flush()
            scheduler.step()
        logs.close()

    ##### TEST
    else:
        net.eval()
        if args.isreduced:
            filelist_test_chose = filelist_test_reduced
        else:
            filelist_test_chose = filelist_test

        for filename in filelist_test_chose:
            print(filename)
            ds = PartDatasetTest(filename, os.path.join(args.rootdir, "test", "pointcloud_npy"),
                            block_size=args.block_size,
                            npoints= args.npoints,
                            test_step=args.test_step,
                            nocolor=args.nocolor
                            )
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.threads
                                                 )

            xyzrgb = ds.xyzrgb[:,:3]  # all points in a file
            scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
            with torch.no_grad():
                t = tqdm(loader, ncols=80)
                for pts, rgb, indices in t:  # indices:torch.Size([16, 8192]) outputs:torch.Size([16, 8192, 8])

                    # pts = pts.transpose(2, 1)
                    # rgb = rgb.transpose(2, 1)  # torch.Size([16, 3, 8192])
                    # gtFeature = getGtFeature(pts)  # torch.Size([16, dim_num, 8192])
                    #
                    # if not args.nocolor:
                    #     feature = torch.cat((gtFeature, rgb), 1)  # 将颜色拼接
                    # else:
                    #     feature = gtFeature

                    feature = pts
                    feature = feature.cuda()
                    outputs = net(feature)

                    outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))  # (batch_size * n_point, 8)
                    scores[indices.cpu().numpy().ravel()] += outputs_np

            mask = np.logical_not(scores.sum(1)==0)
            scores = scores[mask]  # those one not sampled to be discarded
            pts_src = xyzrgb[mask]

            # create the scores for all points
            scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)

            # compute softmax
            scores = scores - scores.max(axis=1)[:, None]
            scores = np.exp(scores) / np.exp(scores).sum(1)[:, None]
            scores = np.nan_to_num(scores)

            os.makedirs(os.path.join(args.savedir, "predicted_label"), exist_ok=True)

            # saving labels
            save_fname = os.path.join(args.savedir, "predicted_label", filename)
            scores = scores.argmax(1)
            np.savetxt(save_fname, scores, fmt='%d')

            if args.savepts:
                save_fname = os.path.join(args.savedir, "predicted_label", f"{filename}_pts.txt")
                xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
                np.savetxt(save_fname, xyzrgb, fmt=['%.4f', '%.4f', '%.4f', '%d'])

            # break

if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
