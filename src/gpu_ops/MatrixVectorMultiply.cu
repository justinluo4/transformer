#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

const int block_size = 512;
const int num_blocks = 50;

struct Shared{
    float row[block_size];
    float in_vec[];
};

template<typename input_float_t>
__global__
void mat_vec_bias_mul_kernel(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out){
    extern __shared__ Shared shared[];
    float *in_vec = shared->in_vec;
    float *row = shared->row;
    int tid = threadIdx.x;
    while(tid < k){
        in_vec[tid] = (float)vec[tid];
        tid += blockDim.x;
    }
    __syncthreads();
    int bid = blockIdx.x;
    while(bid < m){
        tid = threadIdx.x;
        float tsum = 0.0;
        while(tid < k){
            tsum += in_vec[tid] * (float)mat[bid * k + tid];
            tid += blockDim.x;
        }
        tid = threadIdx.x;
        row[tid] = tsum;
        __syncthreads();
        for (int s = blockDim.x/2; s >= 1; s >>= 1){
            if (tid < s){
                row[tid] = row[tid] + row[tid + s];
            }
            __syncthreads();
        }
        
        if(threadIdx.x == 0){
            out[bid] = (__nv_bfloat16) (row[0] + (float)bias[bid]);
        }

        bid += gridDim.x;
    }
}

template<typename input_float_t>
__global__
void mat_vec_mul_kernel(int32_t m, int32_t k, __nv_bfloat16 *mat, input_float_t *vec, __nv_bfloat16 *out){
    extern __shared__ Shared shared[];
    float *in_vec = shared->in_vec;
    float *row = shared->row;
    int tid = threadIdx.x;
    while(tid < k){
        in_vec[tid] = (float)vec[tid];
        tid += blockDim.x;
    }
    __syncthreads();
    int bid = blockIdx.x;
    while(bid < m){
        tid = threadIdx.x;
        float tsum = 0.0;
        while(tid < k){
            tsum += in_vec[tid] * (float)mat[bid * k + tid];
            tid += blockDim.x;
        }
        tid = threadIdx.x;
        row[tid] = tsum;
        __syncthreads();
        for (int s = blockDim.x/2; s >= 1; s >>= 1){
            if (tid < s){
                row[tid] = row[tid] + row[tid + s];
            }
            __syncthreads();
        }
        
        if(threadIdx.x == 0){
            out[bid] = (__nv_bfloat16) row[0];
        }

        bid += gridDim.x;
    }
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {

    int32_t shared_size = block_size * sizeof(float) + k * sizeof(float);

    if(!bias){
        mat_vec_mul_kernel<input_float_t><<<num_blocks, block_size, shared_size, stream>>>(m, k, mat, vec, out);
    }
    else{
        mat_vec_bias_mul_kernel<input_float_t><<<num_blocks, block_size, shared_size, stream>>>(m, k, mat, bias, vec, out);

    }
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
