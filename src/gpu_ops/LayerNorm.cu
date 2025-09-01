#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

const int block_size = 512;
const int num_blocks = 128;


__global__
void rms_kernel(const __nv_bfloat16 *data, float* rms, int32_t n) {
    extern __shared__ float shared[];

    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    float tot = 0.;
    
    // Each thread calculates the max over a subset of the values, then  
    // saves it into the shared memory.
    
    while (thread_index < n) {
        float d = (float) data[thread_index];

        tot += d*d;
        
        thread_index += blockDim.x * gridDim.x;
    }
    
    shared[idx] = tot;
    
    __syncthreads();

    
    for (int s = blockDim.x/2; s >= 1; s >>= 1){
        if (idx < s){
            shared[idx] = shared[idx] + shared[idx + s];
            
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        atomicAdd(rms, shared[0]);
    }
    
}

__global__
void norm_kernel(const __nv_bfloat16 *data, const __nv_bfloat16 *weights, __nv_bfloat16 *output, float rms, int32_t n) {

    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < n) {
        output[thread_index] = (__nv_bfloat16)((float)data[thread_index] * (float)weights[thread_index]/rms);
        
        thread_index += blockDim.x * gridDim.x;
    }

}


LayerNorm::LayerNorm(int32_t len) {
    rms = std::make_shared<CudaBuffer>(sizeof(float));
    
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    float *rms_ptr = (float *)rms->data;
    *rms_ptr = 0.0;
    int32_t n = hidden_state->size/2;
    rms_kernel<<<num_blocks, block_size, block_size*sizeof(float), stream>>>((__nv_bfloat16 *)hidden_state->data, rms_ptr, n);
    cudaStreamSynchronize(stream);
    *rms_ptr = sqrtf(*rms_ptr/n) + EPS;
    norm_kernel<<<num_blocks, block_size, 0, stream>>>((__nv_bfloat16 *)hidden_state->data, (__nv_bfloat16 *)weights->data, (__nv_bfloat16 *)output->data, *rms_ptr, n);
}   
