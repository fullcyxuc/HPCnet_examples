import torch
import torch.nn as nn
from HPCnet.pointnet2_utils.pointnet2_modules import PointnetFPModule, HPC_SAModuleMSG
import etw_pytorch_utils as pt_utils

def get_model(input_channels=0, num_classes=1):
    return Pointnet2MSGSEM(input_channels=input_channels, num_classes=num_classes)


NPOINTS = [4096, 1024, 256]
RADIUS = [[0.4, 0.8], [0.8, 1.2], [1.2, 1.6]]
NSAMPLE = [[16, 32], [16, 32], [16, 32]]
MLPS = [[[45, 45, 64], [45, 45, 64]], [[192, 192, 256], [192, 192, 256]], [[320, 320, 512], [320, 320, 512]]]
FP_MLPS = [[128, 128], [256, 256], [512, 512]]
DP_RATIO = 0.4


class Pointnet2MSGSEM(nn.Module):
    def __init__(self, input_channels=0, num_classes=1):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.num_classes = num_classes
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                HPC_SAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
            )

        self.FC_layer = (
            pt_utils.Seq(128)
                .conv1d(128, bn=True)
                .dropout(DP_RATIO)
                .conv1d(self.num_classes, activation=None)
        )

        # self.softmax = nn.LogSoftmax(dim=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        pred_cls = self.FC_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_classes)
        return pred_cls
