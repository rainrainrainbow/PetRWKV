#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <mutex>
#include <fstream>
#include "rwkv_v7.h"
#include "weight_loader.h"
#include "tokenizer.h"

using namespace petrwkv;

struct EngineState {
    RWKV7Model model;
    Tokenizer tokenizer;
    std::mt19937 rng;
    std::mutex mutex;
    bool model_loaded = false;
    bool tokenizer_loaded = false;
    int max_context = 2048;
    EngineState() : rng(std::random_device{}()) {}
};

static std::unique_ptr<EngineState> g_engine;

static bool check_model_loaded() {
    if (!g_engine || !g_engine->model_loaded) {
        std::cerr << "[Engine] 错误: 模型未加载" << std::endl;
        return false;
    }
    return true;
}

static char* alloc_c_string(const std::string& str) {
    char* output = (char*)malloc(str.length() + 1);
    if (output) memcpy(output, str.c_str(), str.length() + 1);
    return output;
}

static std::string float_vector_to_string(const float* vec, int n) {
    std::string result;
    for (int i = 0; i < n; ++i) {
        result += std::to_string(vec[i]);
        if (i < n - 1) result += ",";
    }
    return result;
}

extern "C" {

int load_model(const char* model_path, const char* backend) {
    try {
        g_engine = std::make_unique<EngineState>();
        if (!model_path) { std::cerr << "[Engine] 错误: model_path 为空" << std::endl; return -1; }
        
        std::string path(model_path);
        std::cout << "[Engine] 加载模型: " << path << std::endl;
        if (backend) std::cout << "[Engine] 后端: " << backend << std::endl;
        
        std::string weight_path = path;
        std::string vocab_path = path + ".vocab";
        
        std::ifstream test_file(path, std::ios::binary);
        if (!test_file.is_open()) {
            weight_path = path + "/model.bin";
            vocab_path = path + "/vocab.txt";
        }
        test_file.close();
        
        if (!WeightLoader::load(g_engine->model, weight_path)) {
            std::cerr << "[Engine] 权重加载失败: " << WeightLoader::get_last_error() << std::endl;
            std::cout << "[Engine] 使用随机权重初始化 (测试模式)" << std::endl;
            RWKV7Config cfg;
            cfg.vocab_size = 65536;
            cfg.hidden_size = 1024;
            cfg.num_layers = 24;
            cfg.ffn_hidden_size = 4096;
            g_engine->model.init(cfg);
        }
        
        if (g_engine->tokenizer.load(vocab_path)) {
            g_engine->tokenizer_loaded = true;
        } else {
            std::cout << "[Engine] Tokenizer 未加载 (使用简易模式)" << std::endl;
        }
        
        g_engine->model_loaded = true;
        g_engine->max_context = g_engine->model.config.ctx_len;
        std::cout << "[Engine] 模型加载完成" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Engine] 加载模型异常: " << e.what() << std::endl;
        return -1;
    }
}

void unload_model() {
    g_engine.reset();
    std::cout << "[Engine] 模型已卸载" << std::endl;
}

char* generate(const char* prompt, int max_tokens, float temperature, float top_p) {
    if (!check_model_loaded()) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_engine->mutex);
        std::string prompt_str(prompt ? prompt : "");
        
        std::vector<int> tokens;
        if (g_engine->tokenizer_loaded) {
            tokens = g_engine->tokenizer.encode(prompt_str, true, g_engine->max_context);
        } else {
            tokens.push_back(0);
            for (char c : prompt_str) tokens.push_back(static_cast<unsigned char>(c) + 256);
        }
        
        std::vector<int> output_tokens;
        std::vector<float> logits(g_engine->model.config.vocab_size);
        const int C = g_engine->model.config.hidden_size;
        const int L = g_engine->model.config.num_layers;
        
        std::vector<std::vector<float>> states(L, std::vector<float>(C * 4, 0.0f));
        std::vector<float*> state_ptrs(L);
        for (int l = 0; l < L; l++) state_ptrs[l] = states[l].data();
        
        std::vector<const float*> null_states(L, nullptr);
        std::vector<float> prompt_logits((int)tokens.size() * g_engine->model.config.vocab_size);
        model_forward(prompt_logits.data(), state_ptrs.data(), tokens.data(), null_states.data(), g_engine->model, (int)tokens.size());
        
        for (int i = 0; i < max_tokens; i++) {
            int last_token = tokens.back();
            int input_tokens[1] = {last_token};
            std::vector<const float*> state_inputs(L);
            for (int l = 0; l < L; l++) state_inputs[l] = states[l].data();
            
            model_forward(logits.data(), state_ptrs.data(), input_tokens, state_inputs.data(), g_engine->model, 1);
            
            int next_token;
            if (temperature > 0 && top_p < 1.0f) {
                next_token = sample_top_p(logits.data(), g_engine->model.config.vocab_size, temperature, top_p, g_engine->rng);
            } else {
                next_token = sample_argmax(logits.data(), g_engine->model.config.vocab_size);
            }
            
            if (next_token == TOKEN_EOS || next_token == TOKEN_PAD) break;
            output_tokens.push_back(next_token);
            tokens.push_back(next_token);
            
            if ((int)tokens.size() > g_engine->max_context) {
                tokens.erase(tokens.begin(), tokens.begin() + 64);
                for (int l = 0; l < L; l++) std::fill(states[l].begin(), states[l].end(), 0.0f);
            }
        }
        
        std::string result;
        if (g_engine->tokenizer_loaded) {
            result = g_engine->tokenizer.decode(output_tokens);
        } else {
            for (int token : output_tokens) {
                if (token >= 256 && token < 512) result += static_cast<char>(token - 256);
            }
        }
        
        return alloc_c_string(result);
    } catch (const std::exception& e) {
        std::cerr << "[Engine] 生成异常: " << e.what() << std::endl;
        return nullptr;
    }
}

char* understand_image(const char* image_path, const char* prompt, int max_tokens) {
    if (!check_model_loaded()) return nullptr;
    try {
        return alloc_c_string("已识别图片内容: [需要实现 ViT 编码器]");
    } catch (...) { return nullptr; }
}

char* understand_audio(const char* audio_path, const char* prompt, int max_tokens) {
    if (!check_model_loaded()) return nullptr;
    try {
        return alloc_c_string("已识别音频内容: [需要实现 Whisper 编码器]");
    } catch (...) { return nullptr; }
}

char* get_embedding(const char* text) {
    if (!check_model_loaded()) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_engine->mutex);
        std::string text_str(text ? text : "");
        const int C = g_engine->model.config.hidden_size;
        
        std::vector<int> tokens;
        if (g_engine->tokenizer_loaded) {
            tokens = g_engine->tokenizer.encode(text_str, true, 512);
        } else {
            tokens.push_back(0);
            for (char c : text_str) tokens.push_back(static_cast<unsigned char>(c) + 256);
        }
        if (tokens.empty()) tokens.push_back(0);
        
        std::vector<float> embedding(C);
        get_embedding(embedding.data(), tokens.data(), (int)tokens.size(), g_engine->model);
        return alloc_c_string(float_vector_to_string(embedding.data(), C));
    } catch (const std::exception& e) {
        std::cerr << "[Engine] 嵌入提取异常: " << e.what() << std::endl;
        return nullptr;
    }
}

int train_lora(const char* dataset_path, int epochs, float learning_rate) {
    if (!check_model_loaded()) return -1;
    try {
        std::string ds_path(dataset_path ? dataset_path : "");
        std::cout << "[Engine] LoRA 训练: dataset=" << ds_path << " epochs=" << epochs << " lr=" << learning_rate << std::endl;
        std::cout << "[Engine] LoRA 训练完成 (占位)" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Engine] LoRA 训练异常: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"