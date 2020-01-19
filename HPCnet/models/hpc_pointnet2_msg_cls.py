import torch
import torch.nn as nn
from HPCnet.pointnet2_utils.pointnet2_modules import HPC_SAModuleMSG, PointnetSAModule
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
        channel_out = 0
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
            channel_in = channel_out

        # 分类任务需要加多一层，把特征通过maxpooling到1维，然后传到fc
        self.SA_modules.append(
            PointnetSAModule(mlp=[channel_out, 256, 512, 1024], use_xyz=True, pool_method='max_pool')
        )


        self.FC_layer = (
            pt_utils.Seq(1024)
                .fc(512, bn=True)
                .dropout(DP_RATIO)
                .fc(256, bn=True)
                .dropout(DP_RATIO)
                .fc(num_classes, activation=None)
        )
        # softmax response映射到0到1 不加的话最后算的loss会一直减少而不是收敛接近0
        self.softmax = nn.LogSoftmax(dim=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        for i in range(len(self.SA_modules)):
            xyz, features = self.SA_modules[i](xyz, features)

        pred_cls = self.FC_layer(features.squeeze(-1))  # (B, NUM_CLASS)
        pred_cls = self.softmax(pred_cls)
        return pred_cls


