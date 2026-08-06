#include "rwkv_v7.h"
#include "ops/rwkv_math.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace petrwkv {

void time_mix_forward(float* out, float* new_state, const float* x, const float* state, const TimeMixWeight& w, const RWKV7Config& cfg, int T) {
    const int C = cfg.hidden_size;
    (void)cfg;
    
    std::vector<float> ln_x(T * C);
    std::vector<float> r_buf(T * C), k_buf(T * C), v_buf(T * C);
    std::vector<float> w_buf(T * C), a_buf(T * C), g_buf(T * C);
    std::vector<float> tmp(C);
    
    for (int t = 0; t < T; t++) {
        const float* xt = &x[t * C];
        float* out_t = &out[t * C];
        float* ln_xt = &ln_x[t * C];
        
        layer_norm(ln_xt, xt, w.ln1.weight.data(), w.ln1.bias.data(), C);
        
        for (int i = 0; i < C; i++) r_buf[t * C + i] = ln_xt[i] * w.x_r[i];
        matmul_bt(tmp.data(), ln_xt, w.receptance.weight.data(), 1, C, C);
        for (int i = 0; i < C; i++) r_buf[t * C + i] += tmp[i];
        
        for (int i = 0; i < C; i++) k_buf[t * C + i] = ln_xt[i] * w.x_k[i];
        matmul_bt(tmp.data(), ln_xt, w.key.weight.data(), 1, C, C);
        for (int i = 0; i < C; i++) k_buf[t * C + i] += tmp[i];
        
        for (int i = 0; i < C; i++) v_buf[t * C + i] = ln_xt[i] * w.x_v[i];
        matmul_bt(tmp.data(), ln_xt, w.value.weight.data(), 1, C, C);
        for (int i = 0; i < C; i++) v_buf[t * C + i] += tmp[i];
        
        std::vector<float> w1_x(cfg.d_decay_lora, 0);
        matmul_bt(w1_x.data(), ln_xt, w.w1.weight.data(), 1, C, cfg.d_decay_lora);
        vec_tanh(w1_x.data(), w1_x.data(), cfg.d_decay_lora);
        std::vector<float> w2_x(C, 0);
        matmul_bt(w2_x.data(), w1_x.data(), w.w2.weight.data(), 1, cfg.d_decay_lora, C);
        for (int i = 0; i < C; i++) w_buf[t * C + i] = w.w0[i] + w2_x[i];
        
        std::vector<float> a1_x(cfg.d_aaa_lora, 0);
        matmul_bt(a1_x.data(), ln_xt, w.a1.weight.data(), 1, C, cfg.d_aaa_lora);
        vec_tanh(a1_x.data(), a1_x.data(), cfg.d_aaa_lora);
        std::vector<float> a2_x(C, 0);
        matmul_bt(a2_x.data(), a1_x.data(), w.a2.weight.data(), 1, cfg.d_aaa_lora, C);
        for (int i = 0; i < C; i++) a_buf[t * C + i] = w.a0[i] + a2_x[i];
        
        std::vector<float> g1_x(cfg.d_gate_lora, 0);
        matmul_bt(g1_x.data(), ln_xt, w.g1.weight.data(), 1, C, cfg.d_gate_lora);
        vec_tanh(g1_x.data(), g1_x.data(), cfg.d_gate_lora);
        std::vector<float> g2_x(C, 0);
        matmul_bt(g2_x.data(), g1_x.data(), w.g2.weight.data(), 1, cfg.d_gate_lora, C);
        vec_sigmoid(g_buf.data() + t * C, g2_x.data(), C);
    }
    
    for (int t = 0; t < T; t++) {
        float* kt = &k_buf[t * C];
        for (int i = 0; i < C; i++) kt[i] *= w.k_k[i];
        vec_normalize(kt, kt, C);
        for (int i = 0; i < C; i++) kt[i] *= w.k_a[i];
    }
    
    std::vector<float> aa(C, 0.0f), bb(C, 0.0f), pp(C, 0.0f), qq(C, 0.0f);
    if (state) {
        memcpy(aa.data(), state, C * sizeof(float));
        memcpy(bb.data(), state + C, C * sizeof(float));
        memcpy(pp.data(), state + C * 2, C * sizeof(float));
        memcpy(qq.data(), state + C * 3, C * sizeof(float));
    }
    
    for (int t = 0; t < T; t++) {
        const float* r_t = &r_buf[t * C];
        const float* k_t = &k_buf[t * C];
        const float* v_t = &v_buf[t * C];
        const float* w_t = &w_buf[t * C];
        float* out_t = &out[t * C];
        
        for (int i = 0; i < C; i++) {
            float ww = sigmoid(w_t[i]);
            aa[i] = aa[i] * ww + k_t[i] * v_t[i];
            bb[i] = bb[i] * ww + k_t[i];
            out_t[i] = r_t[i] * (aa[i] / (bb[i] + 1e-8f));
        }
        memcpy(pp.data(), k_t, C * sizeof(float));
        memcpy(qq.data(), v_t, C * sizeof(float));
    }
    
    if (new_state) {
        memcpy(new_state, aa.data(), C * sizeof(float));
        memcpy(new_state + C, bb.data(), C * sizeof(float));
        memcpy(new_state + C * 2, pp.data(), C * sizeof(float));
        memcpy(new_state + C * 3, qq.data(), C * sizeof(float));
    }
    
    std::vector<float> gn_out(T * C);
    for (int t = 0; t < T; t++) {
        group_norm(gn_out.data() + t * C, &out[t * C], w.ln_x_weight.data(), w.ln_x_bias.data(), C, 32);
    }
    
    for (int t = 0; t < T; t++) {
        matmul_bt(tmp.data(), &gn_out[t * C], w.output.weight.data(), 1, C, C);
        memcpy(&out[t * C], tmp.data(), C * sizeof(float));
        for (int i = 0; i < C; i++) out[t * C + i] *= g_buf[t * C + i];
    }
}

void channel_mix_forward(float* out, const float* x, const ChannelMixWeight& w, const RWKV7Config& cfg, int T) {
    const int C = cfg.hidden_size;
    const int F = cfg.ffn_hidden_size;
    std::vector<float> tmp(F);
    (void)cfg;
    
    for (int t = 0; t < T; t++) {
        const float* xt = &x[t * C];
        float* out_t = &out[t * C];
        
        float* ln_xt = new float[C];
        layer_norm(ln_xt, xt, w.ln2.weight.data(), w.ln2.bias.data(), C);
        
        matmul_bt(tmp.data(), ln_xt, w.key.weight.data(), 1, C, F);
        for (int i = 0; i < F; i++) tmp[i] = relu2(tmp[i]);
        
        matmul_bt(out_t, tmp.data(), w.value.weight.data(), 1, F, C);
        for (int i = 0; i < C; i++) out_t[i] = xt[i] + out_t[i];
        
        delete[] ln_xt;
    }
}

void block_forward(float* out, float* new_state, const float* x, const float* state, const LayerWeight& w, const RWKV7Config& cfg, int T) {
    const int C = cfg.hidden_size;
    std::vector<float> tm_out(T * C);
    time_mix_forward(tm_out.data(), new_state, x, state, w.time_mix, cfg, T);
    
    std::vector<float> res(T * C);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < C; i++) res[t * C + i] = x[t * C + i] + tm_out[t * C + i];
    }
    channel_mix_forward(out, res.data(), w.channel_mix, cfg, T);
}

void model_forward(float* logits, float** new_states, const int* tokens, const float** states, const RWKV7Model& model, int T) {
    const int C = model.config.hidden_size;
    const int L = model.config.num_layers;
    const int V = model.config.vocab_size;
    
    std::vector<float> hidden(T * C);
    for (int t = 0; t < T; t++) {
        int token = tokens[t];
        if (token >= 0 && token < V) {
            memcpy(&hidden[t * C], &model.emb_weight[token * C], C * sizeof(float));
        }
    }
    
    std::vector<float> layer_out(T * C);
    std::vector<std::vector<float>> layer_states;
    if (new_states) layer_states.resize(L, std::vector<float>(C * 4));
    
    for (int l = 0; l < L; l++) {
        const float* state_in = (states && states[l]) ? states[l] : nullptr;
        float* state_out = new_states ? layer_states[l].data() : nullptr;
        block_forward(layer_out.data(), state_out, hidden.data(), state_in, model.layers[l], model.config, T);
        if (l < L - 1) memcpy(hidden.data(), layer_out.data(), T * C * sizeof(float));
    }
    
    for (int t = 0; t < T; t++) {
        std::vector<float> ln_xt(C);
        layer_norm(ln_xt.data(), &layer_out[t * C], model.ln_out.weight.data(), model.ln_out.bias.data(), C);
        matmul_bt(&logits[t * V], ln_xt.data(), model.head.weight.data(), 1, C, V);
    }
    
    if (new_states) {
        for (int l = 0; l < L; l++) {
            if (new_states[l]) memcpy(new_states[l], layer_states[l].data(), C * 4 * sizeof(float));
        }
    }
}

int generate_token(const int* tokens, int num_tokens, const RWKV7Model& model, float temperature, float top_p, std::mt19937& rng) {
    const int V = model.config.vocab_size;
    std::vector<float> logits(num_tokens * V);
    model_forward(logits.data(), nullptr, tokens, nullptr, model, num_tokens);
    const float* last_logits = &logits[(num_tokens - 1) * V];
    if (temperature > 0 && top_p < 1.0f) return sample_top_p(last_logits, V, temperature, top_p, rng);
    else return sample_argmax(last_logits, V);
}

void get_embedding(float* embedding, const int* tokens, int num_tokens, const RWKV7Model& model) {
    const int C = model.config.hidden_size;
    const int V = model.config.vocab_size;
    
    std::vector<float> hidden(num_tokens * C);
    for (int t = 0; t < num_tokens; t++) {
        int token = tokens[t];
        if (token >= 0 && token < V) memcpy(&hidden[t * C], &model.emb_weight[token * C], C * sizeof(float));
    }
    
    for (int l = 0; l < model.config.num_layers; l++) {
        std::vector<float> layer_out(num_tokens * C);
        block_forward(layer_out.data(), nullptr, hidden.data(), nullptr, model.layers[l], model.config, num_tokens);
        if (l < model.config.num_layers - 1) {
            memcpy(hidden.data(), layer_out.data(), num_tokens * C * sizeof(float));
        } else {
            std::vector<float> ln_out(num_tokens * C);
            for (int t = 0; t < num_tokens; t++) {
                layer_norm(&ln_out[t * C], &layer_out[t * C], model.ln_out.weight.data(), model.ln_out.bias.data(), C);
            }
            std::fill(embedding, embedding + C, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
                for (int i = 0; i < C; i++) embedding[i] += ln_out[t * C + i];
            }
            for (int i = 0; i < C; i++) embedding[i] /= num_tokens;
            vec_normalize(embedding, embedding, C);
        }
    }
}

} // namespace petrwkv