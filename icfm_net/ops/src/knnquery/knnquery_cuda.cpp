#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "knnquery_cuda_kernel.h"


void knnquery_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    knnquery_cuda_launcher(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}



/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/


#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) AT_ASSERTM(x.dtype() == t, #x " must be " #t)
#define CHECK_CUDA(x) AT_ASSERTM(x.device().type() == at::Device::Type::CUDA, #x " must be on CUDA")
#define CHECK_INPUT(x, t) CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x)


std::vector<at::Tensor> knn(
    at::Tensor & ref, 
    at::Tensor & query, 
    const int k
    ){

    CHECK_INPUT(ref, at::kFloat);
    CHECK_INPUT(query, at::kFloat);
    int dim = ref.size(0);
    int ref_nb = ref.size(1);
    int query_nb = query.size(1);
    float * ref_dev = ref.data<float>();
    float * query_dev = query.data<float>();
    auto dist = at::empty({ref_nb, query_nb}, query.options().dtype(at::kFloat));
    auto ind = at::empty({k, query_nb}, query.options().dtype(at::kLong));
    float * dist_dev = dist.data<float>();
    long * ind_dev = ind.data<long>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_device(
        ref_dev,
        ref_nb,
        query_dev,
        query_nb,
        dim,
        k,
        dist_dev,
        ind_dev,
        stream
    );

    return {dist.slice(0, 0, k), ind};
}


/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/

void knn_v2(int nsample, at::Tensor & xyz_tensor, at::Tensor & offset_tensor, at::Tensor & idx_tensor, at::Tensor & dist2_tensor)
{
    CHECK_INPUT(xyz_tensor, at::kFloat);
    CHECK_INPUT(dist2_tensor, at::kFloat);
    CHECK_INPUT(offset_tensor, at::kInt);
    CHECK_INPUT(idx_tensor, at::kInt);

    float * xyz_dev = xyz_tensor.data<float>();
    int * offset_dev = offset_tensor.data<int>();
    int * idx_dev = idx_tensor.data<int>();
    float * dist2_dev = dist2_tensor.data<float>();

    int m = xyz_tensor.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_device_v2(m, nsample, xyz_dev, offset_dev, idx_dev, dist2_dev, stream);
}

