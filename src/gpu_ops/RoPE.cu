#include "RoPE.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

const int block_size = 512;
const int num_blocks = 80;

__global__
void rope_kernel(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
    int32_t position_idx, float theta_base){
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    while(bid < num_heads){
        tid = threadIdx.x;
        __nv_bfloat16 *in_row = x + bid * head_dim;
        while(tid < head_dim){
            int32_t theta_idx = tid % (head_dim / 2);
            float theta_idx_frac = (float)(theta_idx) / (float)(head_dim / 2);
            float theta = powf(theta_base, -theta_idx_frac);
            float angle = theta * (float)(position_idx);
            __nv_bfloat16 rotated_val;
            if (tid < head_dim / 2){
                rotated_val = -in_row[tid + head_dim / 2];
            }
            else{
                rotated_val = in_row[tid - head_dim / 2];
            }
            
            shared[tid] = (float)in_row[tid] * cosf(angle) + (float)rotated_val * sinf(angle);
            

            tid += blockDim.x;
        }
        __syncthreads();
        tid = threadIdx.x;
        while(tid < head_dim){
            in_row[tid] = (__nv_bfloat16)shared[tid];
            tid += blockDim.x;
        }
        __syncthreads();
        bid += gridDim.x;
    }
}



void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    rope_kernel<<<num_blocks, block_size, head_dim*sizeof(float), stream>>>(x, num_heads, head_dim, position_idx, theta_base);
}
