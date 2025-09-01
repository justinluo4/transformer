#include "GroupQueryAttention.cuh"
#include "../qwen2/Qwen2Config.h"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

int block_size = 64;
int num_blocks = 2000;
template <Qwen2Size QWEN2_SIZE>
__global__ void sdp_kernel(__nv_bfloat16* queries, __nv_bfloat16* k, float *sdp, int32_t layer_num, int32_t seq_len) {
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    extern __shared__ float shared[];
    int bid = blockIdx.x;
    int m = Qwen2Config::num_query_heads() * seq_len;

    while(bid < m){
        int head_idx = bid / seq_len;
        int sequence_pos = bid % seq_len;
        
        __nv_bfloat16 *head_query = queries + head_idx * Qwen2Config::head_size();
        int32_t key_idx = head_idx * Qwen2Config::num_kv_heads() / Qwen2Config::num_query_heads();
        int el_idx = threadIdx.x;
        float tot = 0.;
        while (el_idx < Qwen2Config::head_size()) {
            __nv_bfloat16 key_el = *(k +
                    sequence_pos * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) +
                    layer_num * Qwen2Config::keys_size() +
                    key_idx * Qwen2Config::head_size() +
                    el_idx
                );

            tot += float(key_el) * float(head_query[el_idx]);

            el_idx += blockDim.x;
        }
        int idx = threadIdx.x;
        shared[idx] = tot;
        __syncthreads();

        
        for (int s = blockDim.x/2; s >= 1; s >>= 1){
            if (idx < s){
                shared[idx] = shared[idx] + shared[idx + s];
                
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){
            
            float scaled_dot_product = shared[0] / sqrtf((float)Qwen2Config::head_size());
            // if (head_idx == 0){
            // printf("(%d, %d): %f\n", head_idx, sequence_pos, scaled_dot_product);

            // }
            sdp[bid] = scaled_dot_product;
        }

        bid += gridDim.x;
    }


}

struct sumMax {
    float sum;
    float denom;
    float max;

    __device__
    void add(sumMax other){
        if(other.max == -INFINITY){
            return;
        }
        if(this->max == -INFINITY){
            this->sum = other.sum;
            this->max = other.max;
            return;
        }
        if(other.max > this->max){
            this->sum = this->sum * expf(this->max - other.max) + other.sum;
            this->denom = this->denom * expf(this->max - other.max) + other.denom;
            this->max = other.max;
        }
        else{
            this->sum = other.sum * expf(other.max - this->max) + this->sum;
            this->denom = other.denom * expf(other.max - this->max) + this->denom;
        }
    }
    __device__
    void print(){
        printf("%f, %f\n", this->sum, this->max);
    }
};


template <Qwen2Size QWEN2_SIZE>
__global__ void gqa_kernel(float* sdp, __nv_bfloat16* v, float *out, int32_t layer_num, int32_t seq_len) {
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    extern __shared__ sumMax gqa_shared[];
    int bid = blockIdx.x;
    int m = Qwen2Config::num_query_heads() * Qwen2Config::value_size();
    
    while(bid < m){
        int head_idx = bid / Qwen2Config::value_size();
        int el_idx = bid % Qwen2Config::value_size();
        int32_t key_idx = head_idx * Qwen2Config::num_kv_heads() / Qwen2Config::num_query_heads();

        float *out_row = out + head_idx * Qwen2Config::value_size();
        sumMax smax = {0.0, 1.0, -INFINITY};
        int32_t sequence_pos = threadIdx.x;
        while (sequence_pos < seq_len) {
            float cur_sdp = sdp[head_idx * seq_len + sequence_pos];
            
            __nv_bfloat16 val_el = *(v +
                    sequence_pos * (Qwen2Config::num_layers() * Qwen2Config::values_size()) +
                    layer_num * Qwen2Config::values_size() +
                    key_idx * Qwen2Config::value_size() +
                    el_idx
                );
            
            smax.add((sumMax){(float)val_el, 1, cur_sdp});
            

            sequence_pos += blockDim.x;
        }
        int idx = threadIdx.x;
        gqa_shared[idx] = smax;

        
        __syncthreads();

        
        for (int s = blockDim.x/2; s >= 1; s >>= 1){
            if (idx < s){
                gqa_shared[idx].add(gqa_shared[idx+s]);
                
            }
            __syncthreads();
            // if(bid == 0){
            //     printf("%d, %d: %f, %f\n", s, idx, gqa_shared[idx].sum, gqa_shared[idx].max);
            // }
        }
        if(idx == 0){
            // gqa_shared[0].print();
            out_row[el_idx] = gqa_shared[0].sum/gqa_shared[0].denom;
        }

        bid += gridDim.x;
    }


}
template<Qwen2Size QWEN2_SIZE>
GroupQueryAttention<QWEN2_SIZE>::GroupQueryAttention(int32_t max_seq_len){
    size_t bdp_size = Qwen2Config::num_query_heads() * max_seq_len * sizeof(float);
    block_dot_products = std::make_shared<CudaBuffer>(bdp_size);
    
}

template<Qwen2Size QWEN2_SIZE>
void GroupQueryAttention<QWEN2_SIZE>::sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {

    
    sdp_kernel<QWEN2_SIZE><<<num_blocks, block_size, block_size*sizeof(float), stream>>>(queries, k_cache, (float*)block_dot_products->data, layer_num, seq_len);
    
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream));
    gqa_kernel<QWEN2_SIZE><<<num_blocks, block_size, block_size*sizeof(sumMax), stream>>>((float*)block_dot_products->data, v_cache, weighted_values, layer_num, seq_len);
    checkCuda(cudaGetLastError());
    cudaStreamSynchronize(stream);
    checkCuda(cudaGetLastError());

}   


template class GroupQueryAttention<QWEN2_0_5B>;
