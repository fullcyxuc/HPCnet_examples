#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "get_hausdorff_dis_gpu.h"
#include "cuda_utils.h"

#define gt_num 42
#define voxel_dim 31
#define dict_grid_num (voxel_dim*voxel_dim*voxel_dim)
#define prior_point_num 10

__global__ void get_hausdorff_dis_kernel_fast(const float *__restrict__ whole_points,
                                              const float *__restrict__ keypoints,
                                              const float *__restrict__ neighbor_points,
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

    // dim3 blocks(DIVUP(point_num*gt_num, THREADS_PER_BLOCK), batch_size);
    // dim3 threads(gt_num, DIVUP(THREADS_PER_BLOCK, gt_num));
    // dim3 blocks(keypoint_num/16, batch_size);
    // dim3 threads(gt_num, 16);
    int batch_idx = blockIdx.y;
    // int point_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int point_idx = blockIdx.x * 16 + threadIdx.y;
    int gt_idx = threadIdx.x;

    printf("blockDim.x %d\n", blockDim.x);
    printf("blockDim.y %d\n", blockDim.y);
    printf("gridDim.x %d\n", gridDim.x);
    printf("gridDim.y %d\n", gridDim.y);
    printf("threadIdx.x %d\n", threadIdx.x);
    printf("threadIdx.y %d\n", threadIdx.y);
    pirntf("blockIdx.x %d\n", blockIdx.x);
    pirntf("blockIdx.y %d\n", blockIdx.y);
    printf("point_idx %d\n", point_idx);
    printf("gt_idx %d\n", gt_idx);
    // keypoints = batch_idx * keypoint_num * 3 + point_idx * 3;
    // whole_points += batch_idx * whole_point_num * 3;
    neighbor_points += batch_idx * keypoint_num * neighbor_point_num * 3 + point_idx * neighbor_point_num * 3;
    features += batch_idx * keypoint_num * gt_num + point_idx * gt_num + gt_idx;
    dis_dicts += gt_idx * dict_grid_num;
    prior_points += gt_idx * prior_point_num * 3;

    // float r2 = radius * radius;
    // float keypoint_x = keypoints[0];
    // float keypoint_y = keypoints[1];
    // float keypoint_z = keypoints[2];

    float to_prior_dis = 0;
    float tmp_dis;
    int xth, yth, zth;
    int i;
    int prior_hash_idx;
    for( i = 0; i < neighbor_point_num; i++ ){
        xth = floor(abs(neighbor_points[i*3 + 0] + radius) / voxel_len);
        yth = floor(abs(neighbor_points[i*3 + 1] + radius) / voxel_len);
        zth = floor(abs(neighbor_points[i*3 + 2] + radius) / voxel_len);
        prior_hash_idx = xth + yth * voxel_dim + zth * voxel_dim * voxel_dim;
        tmp_dis = dis_dicts[prior_hash_idx];
        if( to_prior_dis < tmp_dis ){
            to_prior_dis = tmp_dis;
        }
    }

    float prior_to_dis = 0;
    float min_point_pair_dis;
    int j;
    for( i = 0; i < prior_point_num; i++ ){
        min_point_pair_dis = 99.9;
        for( j = 0; j < neighbor_point_num; j++ ){
            tmp_dis = ( pow(prior_points[i*3 + 0] - neighbor_points[j*3 + 0], 2) +
                        pow(prior_points[i*3 + 1] - neighbor_points[j*3 + 1], 2) +
                        pow(prior_points[i*3 + 2] - neighbor_points[j*3 + 2], 2) );
            if( min_point_pair_dis > tmp_dis ){
                min_point_pair_dis = tmp_dis;
            }
        }
        if( min_point_pair_dis > prior_to_dis ){
            prior_to_dis = min_point_pair_dis;
        }
    }
    prior_to_dis = sqrt(prior_to_dis);

    float hsdf_dis = prior_to_dis > to_prior_dis? prior_to_dis : to_prior_dis;
    *features  = hsdf_dis > radius? 1 : hsdf_dis / radius;
}

void get_hausdorff_dis_kernel_launcher_fast(const float* whole_points, const float* keypoints,
                                            const float*  neighbor_points,
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

    // dim3 blocks(DIVUP(point_num, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);
    // ball_query_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);

    // dim3 blocks(DIVUP(keypoint_num*gt_num, THREADS_PER_BLOCK), batch_size);
    // dim3 threads(gt_num, DIVUP(THREADS_PER_BLOCK, gt_num));
    dim3 blocks(keypoint_num/16, batch_size);
    dim3 threads(gt_num, 16);

    printf("get_hausdorff_dis_kernel_fast\n");
    get_hausdorff_dis_kernel_fast<<<blocks, threads, 0, stream>>>(
        whole_points, keypoints, neighbor_points, features, radius, batch_size, whole_point_num,
        keypoint_num, neighbor_point_num, prior_points, dis_dicts, voxel_len, stream);

    printf("END get_hausdorff_dis_kernel_fast\n");
    cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
