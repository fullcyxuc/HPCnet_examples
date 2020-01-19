#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "get_hausdorff_dis_gpu.h"
#include "cuda_utils.h"

#define gt_num 42
#define voxel_dim 31
#define dict_grid_num (voxel_dim*voxel_dim*voxel_dim)
#define prior_point_num 9

__global__ void get_hausdorff_dis_kernel_fast(const float *__restrict__ neighbor_points,
                                              float *__restrict__ features, float radius,
                                              int batch_size, int whole_point_num,
                                              int keypoint_num, int neighbor_point_num,
                                              const float* __restrict__ prior_points,
                                              const float* __restrict__ dis_dicts,
                                              float voxel_len, cudaStream_t stream){
    // whole_points: B N C
    // keypoints: B M C
    // neighbor_points: B M nsample C
    // prior_points: Nshapes Npoints_per_shape Cxyz
    // dis_dicts: Nshapes Ngrid Cxyz
    // output:
    //     features: batch_size Nshapes point_num

    // dim3 blocks(DIVUP(point_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);
    int batch_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    neighbor_points += batch_idx * keypoint_num * neighbor_point_num * 3 + point_idx * neighbor_point_num * 3;
    features += batch_idx * keypoint_num * gt_num + point_idx * gt_num;

    float to_prior_dis = 0;
    float tmp_dis;
    int xth, yth, zth;
    int i, j;
    int prior_hash_idx;
    float prior_to_dis = 0;
    float min_point_pair_dis = radius;
    float hsdf_dis = radius;
    for(int gt_idx = 0; gt_idx < gt_num; gt_idx++ ){
        to_prior_dis = 0;
        for( i = 0; i < neighbor_point_num; i++ ){
            xth = floor(abs(neighbor_points[i*3 + 0] + radius) / voxel_len);
            yth = floor(abs(neighbor_points[i*3 + 1] + radius) / voxel_len);
            zth = floor(abs(neighbor_points[i*3 + 2] + radius) / voxel_len);
            prior_hash_idx = xth + yth * voxel_dim + zth * voxel_dim * voxel_dim;
            tmp_dis = dis_dicts[gt_idx*dict_grid_num + prior_hash_idx];
            if( to_prior_dis < tmp_dis ){
                to_prior_dis = tmp_dis;
            }
        }

        prior_to_dis = 0;
        for( i = 0; i < prior_point_num; i++ ){
            min_point_pair_dis = 99.9;
            for( j = 0; j < neighbor_point_num; j++ ){
                tmp_dis = ( pow(prior_points[gt_idx*prior_point_num*3 + i*3 + 0]
                                - neighbor_points[j*3 + 0], 2) +
                            pow(prior_points[gt_idx*prior_point_num*3 + i*3 + 1]
                                - neighbor_points[j*3 + 1], 2) +
                            pow(prior_points[gt_idx*prior_point_num*3 + i*3 + 2]
                                - neighbor_points[j*3 + 2], 2) );
                if( min_point_pair_dis > tmp_dis ){
                    min_point_pair_dis = tmp_dis;
                }
            }
            if( min_point_pair_dis > prior_to_dis ){
                prior_to_dis = min_point_pair_dis;
            }
        }
        prior_to_dis = sqrt(prior_to_dis);

        hsdf_dis = (prior_to_dis > to_prior_dis? prior_to_dis : to_prior_dis) / radius;
        features[gt_idx] = (hsdf_dis > 1? 1: hsdf_dis) < 0.1? 0: hsdf_dis;
    }
}

void get_hausdorff_dis_kernel_launcher_fast(const float*  neighbor_points,
                                            float* features, float radius,
                                            int batch_size, int whole_point_num, int keypoint_num,
                                            int neighbor_point_num,
                                            const float* prior_points, const float* dis_dicts,
                                            float voxel_len, cudaStream_t stream){
    // whole_points: B N C
    // keypoints: B N C
    // neighbor_points: B N nsample C
    // prior_points: Nshapes Npoints_per_shape Cxyz
    // dis_dicts: Nshapes N_hash_grid_per_shape Cxyz
    // output:
    //     features: batch_size point_num Nshapes

    cudaError_t err;

    dim3 blocks(DIVUP(keypoint_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    get_hausdorff_dis_kernel_fast<<<blocks, threads, 0, stream>>>(
        neighbor_points, features, radius, batch_size, whole_point_num,
        keypoint_num, neighbor_point_num, prior_points, dis_dicts, voxel_len, stream);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
