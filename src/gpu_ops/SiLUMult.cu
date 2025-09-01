#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"


const int block_size = 512;
const int num_blocks = 80;

__global__
void silu_kernel(__nv_bfloat16 *x, __nv_bfloat16 *y, int32_t n){
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    while (thread_index < n) {
        float fx = (float) x[thread_index];
        float fy = (float) y[thread_index];
        x[thread_index] = (__nv_bfloat16) (fx / (1.0f + expf(-fx)) * fy);
         
        thread_index += blockDim.x * gridDim.x;
    }
}



void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream) {
    silu_kernel<<<num_blocks, block_size, 0, stream>>>((__nv_bfloat16 *)x->data, (__nv_bfloat16 *)y->data, x->size / 2);
}