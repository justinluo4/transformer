#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

template<Qwen2Size QWEN2_SIZE>
class Qwen2Layer {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num), max_seq_len(max_seq_len), input_layernorm(Qwen2Config::hidden_size()), post_attention_layernorm(Qwen2Config::hidden_size()) {
        int queries_size = Qwen2Config::queries_size();
        int hidden_size = Qwen2Config::hidden_size();
        int keys_size = Qwen2Config::keys_size();
        int intermediate_size = Qwen2Config::intermediate_size();
        int values_size = Qwen2Config::values_size();

        
        q_proj_weight = std::make_shared<CudaBuffer>(queries_size * hidden_size * sizeof(__nv_bfloat16));
        q_proj_bias = std::make_shared<CudaBuffer>(queries_size * sizeof(__nv_bfloat16));
        k_proj_weight = std::make_shared<CudaBuffer>(keys_size * hidden_size * sizeof(__nv_bfloat16));
        k_proj_bias = std::make_shared<CudaBuffer>(keys_size * sizeof(__nv_bfloat16));
        v_proj_weight = std::make_shared<CudaBuffer>(values_size * hidden_size * sizeof(__nv_bfloat16));
        v_proj_bias = std::make_shared<CudaBuffer>(values_size * sizeof(__nv_bfloat16));
        o_proj_weight = std::make_shared<CudaBuffer>(hidden_size * queries_size * sizeof(__nv_bfloat16));
        up_proj_weight = std::make_shared<CudaBuffer>(intermediate_size * intermediate_size * sizeof(__nv_bfloat16));
        gate_proj_weight = std::make_shared<CudaBuffer>(intermediate_size * hidden_size * sizeof(__nv_bfloat16));
        down_proj_weight = std::make_shared<CudaBuffer>(hidden_size * intermediate_size * sizeof(__nv_bfloat16));
        out_proj = std::make_shared<CudaBuffer>(hidden_size * sizeof(float));
        q_proj = std::make_shared<CudaBuffer>(queries_size * sizeof(__nv_bfloat16));
        k_proj = std::make_shared<CudaBuffer>(keys_size * sizeof(__nv_bfloat16));
        v_proj = std::make_shared<CudaBuffer>(values_size * sizeof(__nv_bfloat16));
        up_proj = std::make_shared<CudaBuffer>(intermediate_size * sizeof(__nv_bfloat16));
        gate_proj = std::make_shared<CudaBuffer>(intermediate_size * sizeof(__nv_bfloat16));
        silu_prod = std::make_shared<CudaBuffer>(intermediate_size * sizeof(__nv_bfloat16));
        hidden_temp = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));
        hidden_temp2 = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));

    }

    uint32_t layer_num;
    uint32_t max_seq_len;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, intermediate_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)
    GroupQueryAttention<QWEN2_SIZE> gqa{(int32_t)max_seq_len};

    std::shared_ptr<CudaBuffer> out_proj; 
    std::shared_ptr<CudaBuffer> q_proj; 
    std::shared_ptr<CudaBuffer> k_proj; 
    std::shared_ptr<CudaBuffer> v_proj; 
    std::shared_ptr<CudaBuffer> gate_proj;
    std::shared_ptr<CudaBuffer> up_proj;
    std::shared_ptr<CudaBuffer> silu_prod;
    std::shared_ptr<CudaBuffer> hidden_temp;
    std::shared_ptr<CudaBuffer> hidden_temp2;

    
    /**
     * Pass the hidden state through this layer. Modifies the hidden state in-place.
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param hidden_state current hidden state bf16 (hidden_size,)
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        int queries_size = Qwen2Config::queries_size();
        int hidden_size = Qwen2Config::hidden_size();
        int keys_size = Qwen2Config::keys_size();
        int intermediate_size = Qwen2Config::intermediate_size();
        int values_size = Qwen2Config::values_size();

        input_layernorm.normalize_hidden_state(hidden_state, hidden_temp, stream);
        checkCuda(cudaStreamSynchronize(stream));

        MatrixVectorMultiply::bf16_matmul(queries_size, hidden_size, (__nv_bfloat16 *)(q_proj_weight->data), (__nv_bfloat16 *)(q_proj_bias->data), (__nv_bfloat16 *)(hidden_temp->data), (__nv_bfloat16 *)(q_proj->data), stream);
        checkCuda(cudaStreamSynchronize(stream));
        
        RoPE::apply_rope_to_qk((__nv_bfloat16 *)(q_proj->data), Qwen2Config::num_query_heads(), Qwen2Config::head_size(), seq_len-1, Qwen2Config::rope_theta_base(), stream);
        checkCuda(cudaStreamSynchronize(stream));

        
        __nv_bfloat16 *key_row = (__nv_bfloat16 *)(k_cache->data) + (seq_len - 1) * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) + layer_num * Qwen2Config::keys_size();

        MatrixVectorMultiply::bf16_matmul(keys_size, hidden_size, (__nv_bfloat16 *)(k_proj_weight->data), (__nv_bfloat16 *)(k_proj_bias->data), (__nv_bfloat16 *)(hidden_temp->data), key_row, stream);
        checkCuda(cudaStreamSynchronize(stream));

        RoPE::apply_rope_to_qk(key_row, Qwen2Config::num_kv_heads(), Qwen2Config::head_size(), seq_len-1, Qwen2Config::rope_theta_base(), stream);
        checkCuda(cudaStreamSynchronize(stream));

        __nv_bfloat16 *val_row = (__nv_bfloat16 *)(v_cache->data) + (seq_len - 1) * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) + layer_num * Qwen2Config::keys_size();
        MatrixVectorMultiply::bf16_matmul(values_size, hidden_size, (__nv_bfloat16 *)(v_proj_weight->data), (__nv_bfloat16 *)(v_proj_bias->data), (__nv_bfloat16 *)(hidden_temp->data), val_row, stream);
        checkCuda(cudaStreamSynchronize(stream));
        
        gqa.sdpa(static_cast<__nv_bfloat16*>(q_proj->data),
            static_cast<__nv_bfloat16*>(k_cache->data),
            static_cast<__nv_bfloat16*>(v_cache->data),
            static_cast<float*>(out_proj->data),
            layer_num, seq_len, stream);
        checkCuda(cudaStreamSynchronize(stream));

        MatrixVectorMultiply::bf16_matmul(hidden_size, queries_size, (__nv_bfloat16 *)(o_proj_weight->data), (__nv_bfloat16 *)(hidden_state->data), (float *)(out_proj->data), (__nv_bfloat16 *)(hidden_temp->data), stream);
        checkCuda(cudaStreamSynchronize(stream));

        post_attention_layernorm.normalize_hidden_state(hidden_temp, hidden_temp2, stream);
        checkCuda(cudaStreamSynchronize(stream));

        MatrixVectorMultiply::bf16_matmul(intermediate_size, hidden_size, (__nv_bfloat16 *)(gate_proj_weight->data), (__nv_bfloat16 *)nullptr, (__nv_bfloat16 *)(hidden_temp2->data), (__nv_bfloat16 *)(gate_proj->data), stream);
        checkCuda(cudaStreamSynchronize(stream));

        MatrixVectorMultiply::bf16_matmul(intermediate_size, hidden_size, (__nv_bfloat16 *)(up_proj_weight->data), (__nv_bfloat16 *)nullptr, (__nv_bfloat16 *)(hidden_temp2->data), (__nv_bfloat16 *)(up_proj->data), stream);
        checkCuda(cudaStreamSynchronize(stream));

        SiLUMult::silu_mult_in_place(gate_proj, up_proj, stream);
        checkCuda(cudaStreamSynchronize(stream));

        MatrixVectorMultiply::bf16_matmul(hidden_size, intermediate_size, (__nv_bfloat16 *)(down_proj_weight->data), (__nv_bfloat16 *)(hidden_temp->data), (__nv_bfloat16 *)(gate_proj->data), (__nv_bfloat16 *)(hidden_state->data), stream);
        checkCuda(cudaStreamSynchronize(stream));



    }
};
