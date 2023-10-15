#include "../cuda_utils.h"
#include "knnquery_cuda_kernel.h"


__device__ void swap_float(float *x, float *y)
{
    float tmp = *x;
    *x = *y;
    *y = tmp;
}


__device__ void swap_int(int *x, int *y)
{
    int tmp = *x;
    *x = *y;
    *y = tmp;
}


__device__ void reheap(float *dist, int *idx, int k)
{
    int root = 0;
    int child = root * 2 + 1;
    while (child < k)
    {
        if(child + 1 < k && dist[child+1] > dist[child])
            child++;
        if(dist[root] > dist[child])
            return;
        swap_float(&dist[root], &dist[child]);
        swap_int(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}


__device__ void heap_sort(float *dist, int *idx, int k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        swap_float(&dist[0], &dist[i]);
        swap_int(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}


__device__ int get_bt_idx(int idx, const int *offset)
{
    int i = 0;
    while (1)
    {
        if (idx < offset[i])
            break;
        else
            i++;
    }
    return i;
}


__global__ void knnquery_cuda_kernel(int m, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2) {
    // input: xyz (n, 3) new_xyz (m, 3)
    // output: idx (m, nsample) dist2 (m, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx
    if (pt_idx >= m) return;

    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    dist2 += pt_idx * nsample;

    int bt_idx = get_bt_idx(pt_idx, new_offset);    // batch idx
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    float best_dist[100];
    int best_idx[100];
    for(int i = 0; i < nsample; i++){
        best_dist[i] = 1e10;
        best_idx[i] = start;
    }
    for(int i = start; i < end; i++){
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < best_dist[0]){
            best_dist[0] = d2;
            best_idx[0] = i;
            reheap(best_dist, best_idx, nsample);
        }
    }
    heap_sort(best_dist, best_idx, nsample);
    for(int i = 0; i < nsample; i++){
        idx[i] = best_idx[i];
        dist2[i] = best_dist[i];
    }
}


void knnquery_cuda_launcher(int m, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2) {
    // input: new_xyz: (m, 3), xyz: (n, 3), idx: (m, nsample)
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    knnquery_cuda_kernel<<<blocks, threads, 0>>>(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/

// typedef struct _Range {
//     int start, end;
// } Range;

// __device__ Range new_Range(int s, int e) {
//     Range r;
//     r.start = s;
//     r.end = e;
//     return r;
// }

// __device__ void quick_sort(int arr_idx[], float arr_dist2[], const int len) {
//     if (len <= 0)
//         return; // 避免len等於負值時引發段錯誤（Segment Fault）
//     // r[]模擬列表,p為數量,r[p++]為push,r[--p]為pop且取得元素
//     Range r[len];
//     int p = 0;
//     r[p++] = new_Range(0, len - 1);
//     while (p) {
//         Range range = r[--p];
//         if (range.start >= range.end)
//             continue;
//         float mid = arr_dist2[(range.start + range.end) / 2]; // 選取中間點為基準點
//         int left = range.start, right = range.end;
//         do {
//             while (arr_dist2[left] < mid) ++left;   // 檢測基準點左側是否符合要求
//             while (arr_dist2[right] > mid) --right; //檢測基準點右側是否符合要求
//             if (left <= right) {
//                 swap_float(&arr_dist2[left], &arr_dist2[right]);
//                 swap_int(&arr_idx[left], &arr_idx[right]);
//                 left++;
//                 right--;               // 移動指針以繼續
//             }
//         } while (left <= right);
//         if (range.start < right) r[p++] = new_Range(range.start, right);
//         if (range.end > left) r[p++] = new_Range(left, range.end);
//     }
// }

__device__ void arr_insert(int arr_idx[], float arr_dist2[], const int len, int insert_idx, float insert_dist2)
{
    int p_insert = -1;
    for(int i=0; i<len; i++){
        if(insert_dist2 < arr_dist2[i]){
            p_insert = i;
            break;
        }
    }

    if(p_insert != -1){
        for(int i=len-1; i>p_insert; i--){
            arr_idx[i] = arr_idx[i-1];
            arr_dist2[i] = arr_dist2[i-1];
        }
        arr_idx[p_insert] = insert_idx;
        arr_dist2[p_insert] = insert_dist2;
    }
}

__device__ void bubble_sort(int arr_idx[], float arr_dist2[], int len) {
    int i, j;
    for (i = 0; i < len - 1; i++)
        for (j = 0; j < len - 1 - i; j++)
                if (arr_dist2[j] > arr_dist2[j + 1]) {
                    swap_float(&arr_dist2[j], &arr_dist2[j+1]);
                    swap_int(&arr_idx[j], &arr_idx[j+1]);
                }
}


__global__ void knn_cuda_kernel_v2(int m, int nsample, float *xyz, int *offset, int *idx, float *dist2)
{
    // input: xyz (m, 3)
    // output: idx (m, nsample)
    // output: dist2 (m, nsample)
    unsigned int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx
    if (pt_idx >= m) return;

    float *p_xyz = xyz + pt_idx * 3;
    int *p_idx = idx + pt_idx * nsample;
    float *p_dist2 = dist2 + pt_idx * nsample;

    // get batch idx
    unsigned int bt_idx = 0;
    while(1) {
        if (pt_idx < offset[bt_idx]) break;
        else bt_idx++;
    }

    unsigned int start;
    if (bt_idx == 0) start = 0;
    else start = offset[bt_idx - 1];
    unsigned int end = offset[bt_idx];

    float curr_x = p_xyz[0];
    float curr_y = p_xyz[1];
    float curr_z = p_xyz[2];

    float compare_x = 0;
    float compare_y = 0;
    float compare_z = 0;
    float compare_dis = 0;

    for(int i = 0, si; i < nsample; i++){
        si = start+i;
        p_idx[i] = si;

        compare_x = xyz[si*3 + 0];
        compare_y = xyz[si*3 + 1];
        compare_z = xyz[si*3 + 2];
        p_dist2[i] = (curr_x - compare_x) * (curr_x - compare_x) + (curr_y - compare_y) * (curr_y - compare_y) + (curr_z - compare_z) * (curr_z - compare_z);
    }

    // quick_sort(p_idx, p_dist2, nsample);
    bubble_sort(p_idx, p_dist2, nsample);

    start += nsample;
    for(int i = start; i < end; i++){
        compare_x = xyz[i*3 + 0];
        compare_y = xyz[i*3 + 1];
        compare_z = xyz[i*3 + 2];
        compare_dis = (curr_x - compare_x) * (curr_x - compare_x) + (curr_y - compare_y) * (curr_y - compare_y) + (curr_z - compare_z) * (curr_z - compare_z);

        if (compare_dis < p_dist2[nsample-1]){
            arr_insert(p_idx, p_dist2, nsample, i, compare_dis);
        }

        // 只要比其中一个小，就把最大的换掉
        // for(int j = 0; j < nsample; j++){
        //     if(compare_dis < p_dist2[j]){
        //         int max_idx = 0;
        //         float max_dis = p_dist2[0];

        //         for(int k = 1; k < nsample; k++){
        //             if(p_dist2[k] > p_dist2[max_idx]){
        //                 max_idx = k;
        //                 max_dis = p_dist2[k];
        //             }
        //         }

        //         p_idx[max_idx] = i;
        //         p_dist2[max_idx] = compare_dis;

        //         break;
        //     }
        // }
    }
}

void knn_device_v2(int m, int nsample, float *xyz, int *offset, int *idx, float *dist2, cudaStream_t stream)
{
    // input: xyz: (m, 3), idx: (m, nsample), dist2: (m, nsample)
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    knn_cuda_kernel_v2<<<blocks, threads, 0, stream>>>(m, nsample, xyz, offset, idx, dist2);
}

/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/
/************************************************************************************************/


// Includes
#include <cstdio>
#include "cuda.h"

// Constants used by the program
#define BLOCK_DIM                      16
#define DEBUG                          0

/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceGlobal( float* A, int wA,
    float* B, int wB, int dim, float* AB){

  // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
  __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  float tmp;
  float ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A  = BLOCK_DIM * wA;
  step_B  = BLOCK_DIM * wB;
  end_A   = begin_A + (dim-1) * wA;

    // Conditions
  int cond0 = (begin_A + tx < wA); // used to write in shared memory
  int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
  int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix

  // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
    if (a/wA + ty < dim){
      shared_A[ty][tx] = (cond0)? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1)? B[b + wB * ty + tx] : 0;
    }
    else{
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
    if (cond2 && cond1){
      for (int k = 0; k < BLOCK_DIM; ++k){
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp*tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1)
    AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
}


/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist        distance matrix
  * @param ind         index matrix
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
__global__ void cuInsertionSort(float *dist, long *ind, int width, int height, int k){

  // Variables
  int l, i, j;
  float *p_dist;
  long  *p_ind;
  float curr_dist, max_dist;
  long  curr_row,  max_row;
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (xIndex<width){
    // Pointer shift, initialization, and max value
    p_dist   = dist + xIndex;
    p_ind    = ind  + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 1;

    // Part 1 : sort kth firt elementZ
    for (l=1; l<k; l++){
      curr_row  = l * width;
      curr_dist = p_dist[curr_row];
      if (curr_dist<max_dist){
        i=l-1;
        for (int a=0; a<l-1; a++){
          if (p_dist[a*width]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=l; j>i; j--){
          p_dist[j*width] = p_dist[(j-1)*width];
          p_ind[j*width]   = p_ind[(j-1)*width];
        }
        p_dist[i*width] = curr_dist;
        p_ind[i*width]  = l + 1;
      } else {
        p_ind[l*width] = l + 1;
      }
      max_dist = p_dist[curr_row];
    }

    // Part 2 : insert element in the k-th first lines
    max_row = (k-1)*width;
    for (l=k; l<height; l++){
      curr_dist = p_dist[l*width];
      if (curr_dist<max_dist){
        i=k-1;
        for (int a=0; a<k-1; a++){
          if (p_dist[a*width]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=k-1; j>i; j--){
          p_dist[j*width] = p_dist[(j-1)*width];
          p_ind[j*width]   = p_ind[(j-1)*width];
        }
        p_dist[i*width] = curr_dist;
        p_ind[i*width]   = l + 1;
        max_dist             = p_dist[max_row];
      }
    }
  }
}


/**
  * Computes the square root of the first line (width-th first element)
  * of the distance matrix.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param k       number of neighbors to consider
  */
__global__ void cuParallelSqrt(float *dist, int width, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if (xIndex<width && yIndex<k)
    dist[yIndex*width + xIndex] = sqrt(dist[yIndex*width + xIndex]);
}


void debug(float * dist_dev, long * ind_dev, const int query_nb, const int k){
  float* dist_host = new float[query_nb * k];
  long*  idx_host  = new long[query_nb * k];

  // Memory copy of output from device to host
  cudaMemcpy(dist_host, dist_dev,
      query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpy(idx_host, ind_dev,
      query_nb * k * sizeof(long), cudaMemcpyDeviceToHost);

  int i, j;
  for(i = 0; i < k; i++){
    for (j = 0; j < query_nb; j++) {
      if (j % 8 == 0)
        printf("/\n");
      printf("%f ", sqrt(dist_host[i*query_nb + j]));
    }
    printf("\n");
  }
}



//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS                                      //
//-----------------------------------------------------------------------------------------------//

/**
  * K nearest neighbor algorithm
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_nb        number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_nb      number of query points ; width of the matrix
  * @param dim           dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear matrix
  *
  */
void knn_device(float* ref_dev, int ref_nb, float* query_dev, int query_nb,
    int dim, int k, float* dist_dev, long* ind_dev, cudaStream_t stream){

  // Grids ans threads
  dim3 g_16x16(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
  dim3 t_16x16(BLOCK_DIM, BLOCK_DIM, 1);
  if (query_nb % BLOCK_DIM != 0) g_16x16.x += 1;
  if (ref_nb   % BLOCK_DIM != 0) g_16x16.y += 1;
  //
  dim3 g_256x1(query_nb / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (query_nb%256 != 0) g_256x1.x += 1;

  dim3 g_k_16x16(query_nb / BLOCK_DIM, k / BLOCK_DIM, 1);
  dim3 t_k_16x16(BLOCK_DIM, BLOCK_DIM, 1);
  if (query_nb % BLOCK_DIM != 0) g_k_16x16.x += 1;
  if (k  % BLOCK_DIM != 0) g_k_16x16.y += 1;

  // Kernel 1: Compute all the distances
  cuComputeDistanceGlobal<<<g_16x16, t_16x16, 0, stream>>>(ref_dev, ref_nb,
      query_dev, query_nb, dim, dist_dev);

#if DEBUG
  printf("Pre insertionSort\n");
  debug(dist_dev, ind_dev, query_nb, k);
#endif

  // Kernel 2: Sort each column
  cuInsertionSort<<<g_256x1, t_256x1, 0, stream>>>(dist_dev, ind_dev, query_nb, ref_nb, k);

#if DEBUG
  printf("Post insertionSort\n");
  debug(dist_dev, ind_dev, query_nb, k);
#endif

  // Kernel 3: Compute square root of k first elements
  // cuParallelSqrt<<<g_k_16x16,t_k_16x16, 0, stream>>>(dist_dev, query_nb, k);
}




