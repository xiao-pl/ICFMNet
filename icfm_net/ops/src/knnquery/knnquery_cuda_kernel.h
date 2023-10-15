#ifndef _KNNQUERY_CUDA_KERNEL
#define _KNNQUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void knnquery_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor);

void knnquery_cuda_launcher(int m, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2);



/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/


void knn_device_v2(int m, int nsample, float *xyz, int *offset, int *idx, float *dist2, cudaStream_t stream);
void knn_v2(int nsample, at::Tensor & xyz_tensor, at::Tensor & offset_tensor, at::Tensor & idx_tensor, at::Tensor & dist2_tensor);



/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/

std::vector<at::Tensor> knn(
    at::Tensor & ref, 
    at::Tensor & query, 
    const int k
    );

void knn_device(
    float* ref_dev, 
    int ref_nb, 
    float* query_dev, 
    int query_nb, 
    int dim, 
    int k, 
    float* dist_dev, 
    long* ind_dev, 
    cudaStream_t stream
    );

#endif
