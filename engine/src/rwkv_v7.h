#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <memory>
#include <unordered_map>

namespace petrwkv {

struct RWKV7Config {
    int vocab_size = 65536;
    int hidden_size = 1024;
    int num_layers = 24;
    int ffn_hidden_size = 4096;
    int head_size = 64;
    int ctx_len = 2048;
    int d_decay_lora = 64;
    int d_aaa_lora = 64;
    int d_mv_lora = 32;
    int d_gate_lora = 128;
};

struct LinearWeight {
    std::vector<float> weight;
    std::vector<float> bias;
    int in_dim = 0;
    int out_dim = 0;
    LinearWeight() = default;
    LinearWeight(int in_d, int out_d) : in_dim(in_d), out_dim(out_d) {
        weight.resize(in_dim * out_dim, 0.0f);
    }
};

struct LayerNormWeight {
    std::vector<float> weight;
    std::vector<float> bias;
    int dim = 0;
    LayerNormWeight() = default;
    LayerNormWeight(int d) : dim(d) {
        weight.resize(d, 1.0f);
        bias.resize(d, 0.0f);
    }
};

struct TimeMixWeight {
    LayerNormWeight ln1;
    std::vector<float> x_r, x_w, x_k, x_v, x_a, x_g;
    std::vector<float> k_k, k_a, r_k;
    LinearWeight receptance, key, value, output;
    std::vector<float> w0;
    LinearWeight w1, w2;
    std::vector<float> a0;
    LinearWeight a1, a2;
    LinearWeight g1, g2;
    std::vector<float> ln_x_weight, ln_x_bias;
};

struct ChannelMixWeight {
    LayerNormWeight ln2;
    std::vector<float> x_k;
    LinearWeight key, value;
};

struct LayerWeight {
    TimeMixWeight time_mix;
    ChannelMixWeight channel_mix;
};

struct RWKV7Model {
    RWKV7Config config;
    std::vector<float> emb_weight;
    std::vector<LayerWeight> layers;
    LayerNormWeight ln_out;
    LinearWeight head;
    std::vector<std::vector<float>> states;
    bool has_lora = false;
    
    RWKV7Model() = default;
    
    void init(const RWKV7Config& cfg) {
        config = cfg;
        const int C = cfg.hidden_size, F = cfg.ffn_hidden_size, V = cfg.vocab_size;
        emb_weight.resize(V * C, 0.0f);
        layers.resize(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; i++) {
            auto& tm = layers[i].time_mix;
            tm.ln1 = LayerNormWeight(C);
            tm.x_r.resize(C, 0.0f); tm.x_w.resize(C, 0.0f); tm.x_k.resize(C, 0.0f);
            tm.x_v.resize(C, 0.0f); tm.x_a.resize(C, 0.0f); tm.x_g.resize(C, 0.0f);
            tm.k_k.resize(C, 0.71f); tm.k_a.resize(C, 1.02f); tm.r_k.resize(C, 0.0f);
            tm.receptance = LinearWeight(C, C); tm.key = LinearWeight(C, C);
            tm.value = LinearWeight(C, C); tm.output = LinearWeight(C, C);
            tm.w0.resize(C, 0.0f); tm.w1 = LinearWeight(C, cfg.d_decay_lora);
            tm.w2 = LinearWeight(cfg.d_decay_lora, C);
            tm.a0.resize(C, 0.0f); tm.a1 = LinearWeight(C, cfg.d_aaa_lora);
            tm.a2 = LinearWeight(cfg.d_aaa_lora, C);
            tm.g1 = LinearWeight(C, cfg.d_gate_lora); tm.g2 = LinearWeight(cfg.d_gate_lora, C);
            tm.ln_x_weight.resize(C, 1.0f); tm.ln_x_bias.resize(C, 0.0f);
            
            auto& cm = layers[i].channel_mix;
            cm.ln2 = LayerNormWeight(C); cm.x_k.resize(C, 0.0f);
            cm.key = LinearWeight(C, F); cm.value = LinearWeight(F, C);
        }
        ln_out = LayerNormWeight(C);
        head = LinearWeight(C, V);
        states.resize(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; i++) states[i].resize(C * 4, 0.0f);
    }
    
    void reset_state() {
        for (auto& s : states) std::fill(s.begin(), s.end(), 0.0f);
    }
};

void time_mix_forward(float* out, float* new_state, const float* x, const float* state, const TimeMixWeight& w, const RWKV7Config& cfg, int T);
void channel_mix_forward(float* out, const float* x, const ChannelMixWeight& w, const RWKV7Config& cfg, int T);
void block_forward(float* out, float* new_state, const float* x, const float* state, const LayerWeight& w, const RWKV7Config& cfg, int T);
void model_forward(float* logits, float** new_states, const int* tokens, const float** states, const RWKV7Model& model, int T);
int generate_token(const int* tokens, int num_tokens, const RWKV7Model& model, float temperature, float top_p, std::mt19937& rng);
void get_embedding(float* embedding, const int* tokens, int num_tokens, const RWKV7Model& model);

} // namespace petrwkv