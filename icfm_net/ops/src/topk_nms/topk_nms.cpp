/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "topk_nms.h"

#include <numeric>
#include <algorithm>

// Int get_topk_nms(const float *coords, float *heatmap, int *semantic_label, const int *batch_offsets, const int batch_size,
//                     int *sumNPoint_f, int *sumNPoint_b, float radius, float thres, float local_thres, int K,
//                     Int *idxs_f, int maxActive_f, Int *idxs_b, int maxActive_b, Int *topk_idxs, int *sizes){
Int get_topk_nms(const float *coords, float *heatmap, const int *semantic_label, const int *batch_offsets, const int batch_size,
                    float radius, float thres, float local_thres, int K,
                    int maxActive_f, int *topk_idxs, int *sumNPoint_f, Int *idxs_f){
    
    //const int batch_size = at::size(batch_offsets_tensor, 0) - 1;
    //const int batch_size = sizeof(batch_offsets) / sizeof(*batch_offsets) - 1;
    const int K_batch = K / batch_size;
    Int cnt = 0;
    int nPoint = batch_offsets[batch_size];
    Int sorted_idxs[nPoint] = {0}; 
    float *heatmap_cpu;

    heatmap_cpu = new float[nPoint];
    cudaMemcpy(heatmap_cpu, heatmap, nPoint*sizeof(float), cudaMemcpyDeviceToHost);

    //memset(heatmap_tmp, 0, nPoint*sizeof(float));

    //memcpy(heatmap_tmp, heatmap, nPoint * sizeof(double));
    std::iota(sorted_idxs, sorted_idxs+nPoint, 0);

    for(Int batch=0; batch<batch_size; batch++){
        // std::cout<< "pass" << std::endl;
        const int start = batch_offsets[batch];
        const int end = batch_offsets[batch+1];

        Int batch_cnt = 0;
        Int nPoint_batch = end - start ;
        int visited[nPoint_batch] = {0};
        //Int sorted_idx[nPoint_batch] = {1}; 

        //std::cout<< nPoint << std::endl;
        //std::cout<< "start1" << std::endl;

        std::sort(sorted_idxs+start, sorted_idxs+end, 
                [&heatmap_cpu](Int a,Int b){ return heatmap_cpu[a]>heatmap_cpu[b]; });
        //double min_heat = heatmap[sorted_idxs[end]];
        //std::for_each(heatmap_tmp+start, heatmap_tmp+end, [min_heat](int x){x /= min_heat;} );
        //printf("min_heat: %f", min_heat);

        //std::cout<< sorted_idx << std::endl;
        for(Int i = start; i < end; i++){
            Int idx = sorted_idxs[i];
            if (visited[idx - start] == 0){
                float max_conf = heatmap_cpu[idx];
                if (max_conf < thres) break;
                //printf("idx: %d, score: %f", idx, heatmap[idx]);
                Int tmp_idxs_f[maxActive_f] = {0};
                // Int tmp_idxs_b[maxActive_b] = {0};
                int tmp_maxActive_f = maxActive_f;
                // int tmp_maxActive_b = maxActive_b;

                cudaStream_t stream = at::cuda::getCurrentCUDAStream();

                // 查找以 idx 为中心，R 范围内的同类点，2R 范围内的非同类点，返回同类点个数
                int size = describe_candidates_cuda(idx, heatmap, coords, semantic_label, nPoint,
                                                local_thres, start, end, radius, visited,
                                                tmp_idxs_f, &tmp_maxActive_f, stream);
                //printf("idx: %d, score: %f", idx, heatmap[idx]);
                for (Int f = 0; f < tmp_maxActive_f; f++){
                    if (max_conf < heatmap_cpu[tmp_idxs_f[f]]) max_conf = heatmap_cpu[tmp_idxs_f[f]];
                }
                if ((heatmap_cpu[idx] / max_conf) < local_thres) size = 0; 

                if(size != 0){
                    for (Int j = 0; j < tmp_maxActive_f; j ++){
                        //idxs_f[(*sumNPoint_f + j) * 2 + 0] = tmp_idxs_f[j * 2 + 0];
                        idxs_f[(*sumNPoint_f + j) * 2 + 0] = cnt + batch_cnt;
                        idxs_f[(*sumNPoint_f + j) * 2 + 1] = tmp_idxs_f[j];
                    } 
                    *sumNPoint_f += tmp_maxActive_f;

                    // for (Int k = 0; k < tmp_maxActive_b; k ++){
                    //     idxs_b[(*sumNPoint_b + k) * 2 + 0] = cnt + batch_cnt;
                    //     idxs_b[(*sumNPoint_b + k) * 2 + 1] = tmp_idxs_b[k];
                    // } 
                    // *sumNPoint_b += tmp_maxActive_b;

                    topk_idxs[cnt + batch_cnt] = idx;
                    // sizes[cnt + batch_cnt] = tmp_maxActive_f;
                    batch_cnt ++;
                }
                if (batch_cnt >= K_batch) break;
            }
        }
        cnt += batch_cnt; 
    }
    free(heatmap_cpu);
    return cnt;
}

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) AT_ASSERTM(x.dtype() == t, #x " must be " #t)
#define CHECK_CUDA(x) AT_ASSERTM(x.device().type() == at::Device::Type::CUDA, #x " must be on CUDA")
#define CHECK_INPUT(x, t) CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x)

// Int topk_nms(at::Tensor coords_tensor, at::Tensor heatmap_tensor, at::Tensor batch_offsets_tensor,
// at::Tensor semantic_labels_tensor, float R, float thres, float local_thres, int K, at::Tensor topk_idxs_tensor, 
// at::Tensor sizes_tensor, at::Tensor k_foreground_tensor, int maxActive_f, at::Tensor k_background_tensor, int maxActive_b){
Int topk_nms(
    at::Tensor coords_tensor, at::Tensor heatmap_tensor, at::Tensor batch_offsets_tensor, at::Tensor semantic_labels_tensor,
    float R, float thres, float local_thres, int K, at::Tensor topk_idxs_tensor, int maxActive_f, at::Tensor k_foreground_tensor)
{
    CHECK_INPUT(coords_tensor, at::kFloat);
    CHECK_INPUT(heatmap_tensor, at::kFloat);
    CHECK_TYPE(batch_offsets_tensor, at::kInt);
    CHECK_INPUT(semantic_labels_tensor, at::kInt);
    CHECK_TYPE(topk_idxs_tensor, at::kInt);

    const float *coords = coords_tensor.data<float>();
    float *heatmap = heatmap_tensor.data<float>();
    const int *batch_offsets = batch_offsets_tensor.data<int>();
    const int batch_size = at::size(batch_offsets_tensor, 0) - 1;
    int *semantic_label = semantic_labels_tensor.data<int>();
    int *topk_idxs = topk_idxs_tensor.data<int>();
    // int *sizes = sizes_tensor.data<int>();
    Int *k_foreground = k_foreground_tensor.data<Int>();
    // Int *k_background = k_background_tensor.data<Int>();

    int sumNPoint_f = 0;
    // int sumNPoint_b = 0;

    Int cnt = get_topk_nms(coords, heatmap, semantic_label, batch_offsets, batch_size,
                            R, thres, local_thres, K, maxActive_f, topk_idxs, &sumNPoint_f, k_foreground);

    k_foreground_tensor.resize_({sumNPoint_f, 2});
    // k_background_tensor.resize_({sumNPoint_b, 2});
    
    return cnt;     // 有效的候选点数
}



