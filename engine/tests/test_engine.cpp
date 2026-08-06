#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "rwkv_v7.h"
#include "ops/rwkv_math.h"

using namespace petrwkv;

int test_count = 0;
int pass_count = 0;

#define TEST(name) do { test_count++; std::cout << "  [测试] " << name << "... "; } while(0)
#define PASS() do { pass_count++; std::cout << "\u2705 PASS" << std::endl; } while(0)
#define FAIL(msg) do { std::cout << "\u274c FAIL: " << msg << std::endl; } while(0)

void test_math_ops() {
    std::cout << "\n=== 数学运算测试 ===" << std::endl;
    
    TEST("sigmoid(0) = 0.5");
    assert(std::abs(sigmoid(0.0f) - 0.5f) < 1e-5f); PASS();
    
    TEST("sigmoid large positive ~ 1");
    assert(std::abs(sigmoid(10.0f) - 1.0f) < 1e-4f); PASS();
    
    TEST("sigmoid large negative ~ 0");
    assert(std::abs(sigmoid(-10.0f) - 0.0f) < 1e-4f); PASS();
    
    TEST("gelu(0) = 0");
    assert(std::abs(gelu(0.0f)) < 1e-5f); PASS();
    
    TEST("relu2(3) = 9");
    assert(std::abs(relu2(3.0f) - 9.0f) < 1e-5f); PASS();
    
    TEST("relu2(-3) = 0");
    assert(std::abs(relu2(-3.0f)) < 1e-5f); PASS();
    
    TEST("dot_product");
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    assert(std::abs(dot_product(a, b, 3) - 32.0f) < 1e-5f); PASS();
    
    TEST("layer_norm");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    layer_norm(out, x, w, bias, 4);
    float mean = 0, var = 0;
    for (int i = 0; i < 4; i++) mean += out[i];
    mean /= 4;
    for (int i = 0; i < 4; i++) var += (out[i] - mean) * (out[i] - mean);
    var /= 4;
    assert(std::abs(mean) < 1e-5f);
    assert(std::abs(var - 1.0f) < 1e-4f); PASS();
    
    TEST("softmax");
    float logits[] = {1.0f, 2.0f, 3.0f};
    float probs[3];
    softmax(probs, logits, 3);
    float sum = 0;
    for (int i = 0; i < 3; i++) sum += probs[i];
    assert(std::abs(sum - 1.0f) < 1e-5f);
    assert(probs[2] > probs[1] && probs[1] > probs[0]); PASS();
    
    TEST("sample_argmax");
    float vals[] = {0.1f, 0.5f, 0.3f, 0.8f, 0.2f};
    assert(sample_argmax(vals, 5) == 3); PASS();
}

void test_model_config() {
    std::cout << "\n=== 模型配置测试 ===" << std::endl;
    
    RWKV7Config cfg;
    cfg.vocab_size = 65536;
    cfg.hidden_size = 1024;
    cfg.num_layers = 24;
    cfg.ffn_hidden_size = 4096;
    
    RWKV7Model model;
    model.init(cfg);
    
    TEST("模型初始化: 层数");
    assert(model.layers.size() == 24); PASS();
    
    TEST("模型初始化: 嵌入维度");
    assert(model.emb_weight.size() == (size_t)65536 * 1024); PASS();
    
    TEST("模型初始化: head 维度");
    assert(model.head.weight.size() == (size_t)1024 * 65536); PASS();
    
    TEST("模型初始化: 状态大小");
    assert(model.states.size() == 24);
    assert(model.states[0].size() == (size_t)1024 * 4); PASS();
    
    TEST("TimeMix LoRA 权重初始化");
    auto& tm = model.layers[0].time_mix;
    assert(tm.w1.weight.size() == (size_t)1024 * 64);
    assert(tm.a1.weight.size() == (size_t)1024 * 64);
    assert(tm.g1.weight.size() == (size_t)1024 * 128); PASS();
    
    TEST("ChannelMix 权重初始化");
    auto& cm = model.layers[0].channel_mix;
    assert(cm.key.weight.size() == (size_t)1024 * 4096);
    assert(cm.value.weight.size() == (size_t)4096 * 1024); PASS();
    
    TEST("重置状态");
    model.reset_state();
    bool all_zero = true;
    for (size_t i = 0; i < model.states[0].size(); i++) {
        if (model.states[0][i] != 0.0f) { all_zero = false; break; }
    }
    assert(all_zero); PASS();
}

void test_forward() {
    std::cout << "\n=== 前向传播测试 ===" << std::endl;
    
    RWKV7Config cfg;
    cfg.vocab_size = 65536;
    cfg.hidden_size = 1024;
    cfg.num_layers = 24;
    cfg.ffn_hidden_size = 4096;
    
    RWKV7Model model;
    model.init(cfg);
    
    TEST("time_mix_forward");
    {
        TimeMixWeight tm_w;
        tm_w.ln1 = LayerNormWeight(1024);
        tm_w.x_r.resize(1024, 0.01f);
        tm_w.x_k.resize(1024, 0.01f);
        tm_w.x_v.resize(1024, 0.01f);
        tm_w.k_k.resize(1024, 0.71f);
        tm_w.k_a.resize(1024, 1.02f);
        tm_w.w0.resize(1024, 0.0f);
        tm_w.a0.resize(1024, 0.0f);
        tm_w.receptance = LinearWeight(1024, 1024);
        tm_w.key = LinearWeight(1024, 1024);
        tm_w.value = LinearWeight(1024, 1024);
        tm_w.output = LinearWeight(1024, 1024);
        tm_w.w1 = LinearWeight(1024, 64);
        tm_w.w2 = LinearWeight(64, 1024);
        tm_w.a1 = LinearWeight(1024, 64);
        tm_w.a2 = LinearWeight(64, 1024);
        tm_w.g1 = LinearWeight(1024, 128);
        tm_w.g2 = LinearWeight(128, 1024);
        tm_w.ln_x_weight.resize(1024, 1.0f);
        tm_w.ln_x_bias.resize(1024, 0.0f);
        
        std::vector<float> input(1024, 0.5f);
        std::vector<float> output(1024);
        std::vector<float> new_state(1024 * 4);
        
        time_mix_forward(output.data(), new_state.data(), input.data(), nullptr, tm_w, cfg, 1);
        
        bool has_nan = false;
        for (int i = 0; i < 1024; i++) {
            if (std::isnan(output[i]) || std::isinf(output[i])) { has_nan = true; break; }
        }
        assert(!has_nan); PASS();
    }
    
    TEST("channel_mix_forward");
    {
        ChannelMixWeight cm_w;
        cm_w.ln2 = LayerNormWeight(1024);
        cm_w.x_k.resize(1024, 0.01f);
        cm_w.key = LinearWeight(1024, 4096);
        cm_w.value = LinearWeight(4096, 1024);
        
        std::vector<float> input(1024, 0.5f);
        std::vector<float> output(1024);
        channel_mix_forward(output.data(), input.data(), cm_w, cfg, 1);
        
        bool has_nan = false;
        for (int i = 0; i < 1024; i++) {
            if (std::isnan(output[i]) || std::isinf(output[i])) { has_nan = true; break; }
        }
        assert(!has_nan); PASS();
    }
    
    TEST("model_forward (完整推理链)");
    {
        int tokens[] = {0, 100, 200, 300, 400};
        std::vector<float> logits(5 * 65536);
        model_forward(logits.data(), nullptr, tokens, nullptr, model, 5);
        
        bool has_nan = false;
        for (int i = 0; i < 5 * 65536; i++) {
            if (std::isnan(logits[i]) || std::isinf(logits[i])) { has_nan = true; break; }
        }
        assert(!has_nan); PASS();
    }
    
    TEST("generate_token (采样)");
    {
        int tokens[] = {0, 100, 200};
        std::mt19937 rng(42);
        int token = generate_token(tokens, 3, model, 0.8f, 0.9f, rng);
        assert(token >= 0 && token < 65536); PASS();
    }
    
    TEST("get_embedding");
    {
        int tokens[] = {0, 100, 200, 300};
        std::vector<float> emb(1024);
        get_embedding(emb.data(), tokens, 4, model);
        float norm = 0;
        for (int i = 0; i < 1024; i++) norm += emb[i] * emb[i];
        norm = std::sqrt(norm);
        assert(std::abs(norm - 1.0f) < 1e-4f); PASS();
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  PetRWKV Engine \u5355\u5143\u6d4b\u8bd5" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_math_ops();
    test_model_config();
    test_forward();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  \u6d4b\u8bd5\u7ed3\u679c: " << pass_count << "/" << test_count << " \u901a\u8fc7" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (pass_count == test_count) ? 0 : 1;
}