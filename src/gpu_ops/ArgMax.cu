#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

const int block_size = 512;
const int num_blocks = 128;

struct argStruct {        
    float val;       
    int32_t ind;   

    __device__
    argStruct comp(argStruct other){
        if (other.val > this->val){
            return other;
        }
        return *this;
    }
    __device__
    void print(){
        printf("%f, %d\n", this->val, this->ind);
    }
};

// __device__ static unsigned long long int __argStruct_as_longlong(argStruct v){
//     return *(unsigned long long int *)((void*) &v);
// }

// __device__ static argStruct __longlong_as_argStruct(unsigned long long int i){
//     return *(argStruct *)((void*) &i);
// }

// __device__ static argStruct atomicArgMax(argStruct* address, argStruct val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = ::atomicCAS(address_as_ull, assumed,
//             __argStruct_as_longlong(
//                             __longlong_as_argStruct(assumed).comp(val)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);
//     return __longlong_as_argStruct(old);
// }

__global__ 
void collect_kernel(argStruct *scratch, int32_t *result, int32_t n){
    int idx = threadIdx.x;
    for (int s = blockDim.x/2; s >= 1; s >>= 1){
        if (idx < s){
            scratch[idx] = scratch[idx].comp(scratch[idx + s]);
        }
        __syncthreads();
    }
    *result = scratch[0].ind;

}


__global__
void argmax_kernel(const __nv_bfloat16 *data, argStruct *scratch, argStruct *max_val, int32_t n) {
    __shared__ argStruct block_max[1024];

    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    argStruct max = {-std::numeric_limits<float>::infinity(), -1};
    // Each thread calculates the max over a subset of the values, then  
    // saves it into the shared memory.
    while (thread_index < n) {
        argStruct cur = {(float)data[thread_index], (int)thread_index};

        max = max.comp(cur);
        
        thread_index += blockDim.x * gridDim.x;
    }
    
    block_max[idx] = max;
    __syncthreads();

    for (int s = blockDim.x/2; s >= 1; s >>= 1){
        if (idx < s){
            block_max[idx] = block_max[idx].comp(block_max[idx + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        // block_max[0].print();
        // atomicArgMax(max_val, block_max[0]);
        scratch[blockIdx.x] = block_max[0];
    }

}

ArgMax::ArgMax(int32_t len) {
    temp_space = std::make_shared<CudaBuffer>(num_blocks * sizeof(argStruct));
    dev_max_val = std::make_shared<CudaBuffer>(sizeof(argStruct));
    ans = std::make_shared<CudaBuffer>(sizeof(int32_t));

}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {

    
    argStruct *max_ptr = (argStruct *)dev_max_val->data;
    *max_ptr = {-std::numeric_limits<float>::infinity(), -1};
    argmax_kernel<<<num_blocks, block_size, 0, stream>>>((__nv_bfloat16 *)bf16_data->data, (argStruct *)temp_space->data, (argStruct * )dev_max_val->data, bf16_data->size/2);
    checkCuda(cudaStreamSynchronize(stream));
    collect_kernel<<<1, num_blocks, 0, stream>>>((argStruct *)temp_space->data, (int32_t * )ans->data, num_blocks);
    checkCuda(cudaStreamSynchronize(stream));
    return (int32_t *)ans->data;
}
