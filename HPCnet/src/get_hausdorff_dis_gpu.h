#ifndef _GET_HAUSDORFF_DIS_GPU_H
#define _GET_HAUSDORFF_DIS_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int get_hausdorff_dis_wrapper_fast(at::Tensor neighbor_points_tensor,
                                   at::Tensor features_tensor, float radius,
                                   int batch_size, int whole_point_num,
                                   int keypoint_num, int neighbor_point_num,
                                   at::Tensor prior_points_tensor, at::Tensor dis_dicts_tensor,
                                   float voxel_len);

void get_hausdorff_dis_kernel_launcher_fast(const float*  neighbor_points,
                                            float* features, float radius,
                                            int batch_size, int whole_point_num,
                                            int keypoint_num, int neighbor_point_num,
                                            const float* prior_points, const float* dis_dicts,
                                            float voxel_len, cudaStream_t stream);

#endif
