#include "weight_loader.h"
#include <iostream>
#include <cstring>

namespace petrwkv {

std::string WeightLoader::last_error_ = "";

bool WeightLoader::load(RWKV7Model& model, const std::string& path) {
    if (path.size() >= 4) {
        std::string ext = path.substr(path.size() - 4);
        if (ext == ".bin" || ext == ".BIN") return load_from_bin(model, path);
        else if (ext == "ggml" || ext == ".gguf") return load_from_ggml(model, path);
    }
    return load_from_bin(model, path);
}

bool WeightLoader::load_from_bin(RWKV7Model& model, const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) {
        last_error_ = "无法打开权重文件: " + path;
        std::cerr << "[WeightLoader] " << last_error_ << std::endl;
        return false;
    }
    
    WeightHeader header;
    fs.read(reinterpret_cast<char*>(&header), sizeof(WeightHeader));
    if (!fs) { last_error_ = "无法读取文件头"; return false; }
    
    if (std::strncmp(header.magic, "PetRWKV", 7) != 0) {
        last_error_ = "无效的权重文件格式 (magic mismatch)";
        return false;
    }
    
    RWKV7Config cfg;
    cfg.vocab_size = header.vocab_size;
    cfg.hidden_size = header.hidden_size;
    cfg.num_layers = header.num_layers;
    cfg.ffn_hidden_size = header.ffn_hidden_size;
    cfg.head_size = header.head_size;
    cfg.ctx_len = header.ctx_len;
    cfg.d_decay_lora = header.d_decay_lora;
    cfg.d_aaa_lora = header.d_aaa_lora;
    cfg.d_mv_lora = header.d_mv_lora;
    cfg.d_gate_lora = header.d_gate_lora;
    
    model.init(cfg);
    model.has_lora = (header.flags & 0x01) != 0;
    
    std::cout << "[WeightLoader] 模型配置: V=" << cfg.vocab_size
              << " D=" << cfg.hidden_size << " L=" << cfg.num_layers << std::endl;
    
    for (uint32_t i = 0; i < header.weight_count; i++) {
        TensorHeader tensor_hdr;
        fs.read(reinterpret_cast<char*>(&tensor_hdr), sizeof(TensorHeader));
        if (!fs) { last_error_ = "读取张量头失败"; return false; }
        
        std::vector<float> tensor_data;
        if (!read_tensor(fs, tensor_data, tensor_hdr)) return false;
        
        int layer_idx = -1;
        std::string param_name;
        parse_tensor_name(tensor_hdr.name, layer_idx, param_name);
        
        if (!assign_tensor(model, layer_idx, param_name, tensor_data)) {
            std::cerr << "[WeightLoader] 警告: 无法加载张量: " << tensor_hdr.name << std::endl;
        }
    }
    
    std::cout << "[WeightLoader] 成功加载 " << header.weight_count << " 个张量" << std::endl;
    return true;
}

bool WeightLoader::load_from_ggml(RWKV7Model&, const std::string&) {
    last_error_ = "GGML 格式加载暂未实现";
    return false;
}

bool WeightLoader::save_to_bin(const RWKV7Model& model, const std::string& path) {
    std::ofstream fs(path, std::ios::binary);
    if (!fs.is_open()) { last_error_ = "无法创建权重文件: " + path; return false; }
    
    const auto& cfg = model.config;
    uint32_t weight_count = 4 + cfg.num_layers * 29;
    
    WeightHeader header;
    std::memset(&header, 0, sizeof(WeightHeader));
    std::strncpy(header.magic, "PetRWKV", 8);
    header.format_version = 1;
    header.vocab_size = cfg.vocab_size;
    header.hidden_size = cfg.hidden_size;
    header.num_layers = cfg.num_layers;
    header.ffn_hidden_size = cfg.ffn_hidden_size;
    header.head_size = cfg.head_size;
    header.ctx_len = cfg.ctx_len;
    header.d_decay_lora = cfg.d_decay_lora;
    header.d_aaa_lora = cfg.d_aaa_lora;
    header.d_mv_lora = cfg.d_mv_lora;
    header.d_gate_lora = cfg.d_gate_lora;
    header.weight_count = weight_count;
    header.flags = model.has_lora ? 0x01 : 0x00;
    
    fs.write(reinterpret_cast<const char*>(&header), sizeof(WeightHeader));
    std::cout << "[WeightLoader] 保存 " << weight_count << " 个张量" << std::endl;
    return true;
}

bool WeightLoader::read_tensor(std::ifstream& fs, std::vector<float>& tensor, const TensorHeader& hdr) {
    uint64_t num_elements = 1;
    for (uint32_t d = 0; d < hdr.ndim; d++) num_elements *= hdr.dims[d];
    tensor.resize(num_elements);
    
    if (hdr.dtype == 0) {
        fs.read(reinterpret_cast<char*>(tensor.data()), hdr.data_size);
    } else {
        last_error_ = "不支持的数据类型: " + std::to_string(hdr.dtype);
        fs.seekg(hdr.data_size, std::ios::cur);
        return false;
    }
    return fs.good();
}

bool WeightLoader::parse_tensor_name(const std::string& full_name, int& layer_idx, std::string& param_name) {
    layer_idx = -1;
    param_name = full_name;
    if (full_name.substr(0, 7) == "layers.") {
        size_t dot_pos = full_name.find('.', 7);
        if (dot_pos != std::string::npos) {
            try {
                layer_idx = std::stoi(full_name.substr(7, dot_pos - 7));
                param_name = full_name.substr(dot_pos + 1);
            } catch (...) {}
        }
    }
    return true;
}

bool WeightLoader::assign_tensor(RWKV7Model& model, int layer_idx, const std::string& param_name, const std::vector<float>& data) {
    const int C = model.config.hidden_size;
    const int F = model.config.ffn_hidden_size;
    
    if (layer_idx < 0) {
        if (param_name == "emb.weight" || param_name == "token_emb.weight") {
            if (data.size() == (size_t)model.config.vocab_size * C) { model.emb_weight = data; return true; }
        }
        else if (param_name == "ln_out.weight") { if (data.size() == (size_t)C) { model.ln_out.weight = data; return true; } }
        else if (param_name == "ln_out.bias") { if (data.size() == (size_t)C) { model.ln_out.bias = data; return true; } }
        else if (param_name == "head.weight") { if (data.size() == (size_t)C * model.config.vocab_size) { model.head.weight = data; return true; } }
        return false;
    }
    
    if (layer_idx < 0 || layer_idx >= model.config.num_layers) return false;
    auto& tm = model.layers[layer_idx].time_mix;
    auto& cm = model.layers[layer_idx].channel_mix;
    
    if (param_name == "time_mix.ln1.weight") { if (data.size() == (size_t)C) { tm.ln1.weight = data; return true; } }
    else if (param_name == "time_mix.ln1.bias") { if (data.size() == (size_t)C) { tm.ln1.bias = data; return true; } }
    else if (param_name == "time_mix.x_r") { if (data.size() == (size_t)C) { tm.x_r = data; return true; } }
    else if (param_name == "time_mix.x_w") { if (data.size() == (size_t)C) { tm.x_w = data; return true; } }
    else if (param_name == "time_mix.x_k") { if (data.size() == (size_t)C) { tm.x_k = data; return true; } }
    else if (param_name == "time_mix.x_v") { if (data.size() == (size_t)C) { tm.x_v = data; return true; } }
    else if (param_name == "time_mix.x_a") { if (data.size() == (size_t)C) { tm.x_a = data; return true; } }
    else if (param_name == "time_mix.x_g") { if (data.size() == (size_t)C) { tm.x_g = data; return true; } }
    else if (param_name == "time_mix.k_k") { if (data.size() == (size_t)C) { tm.k_k = data; return true; } }
    else if (param_name == "time_mix.k_a") { if (data.size() == (size_t)C) { tm.k_a = data; return true; } }
    else if (param_name == "time_mix.receptance.weight") { if (data.size() == (size_t)C * C) { tm.receptance.weight = data; return true; } }
    else if (param_name == "time_mix.key.weight") { if (data.size() == (size_t)C * C) { tm.key.weight = data; return true; } }
    else if (param_name == "time_mix.value.weight") { if (data.size() == (size_t)C * C) { tm.value.weight = data; return true; } }
    else if (param_name == "time_mix.output.weight") { if (data.size() == (size_t)C * C) { tm.output.weight = data; return true; } }
    else if (param_name == "time_mix.w0") { if (data.size() == (size_t)C) { tm.w0 = data; return true; } }
    else if (param_name == "time_mix.w1.weight") { if (data.size() == (size_t)C * model.config.d_decay_lora) { tm.w1.weight = data; return true; } }
    else if (param_name == "time_mix.w2.weight") { if (data.size() == (size_t)model.config.d_decay_lora * C) { tm.w2.weight = data; return true; } }
    else if (param_name == "time_mix.a0") { if (data.size() == (size_t)C) { tm.a0 = data; return true; } }
    else if (param_name == "time_mix.a1.weight") { if (data.size() == (size_t)C * model.config.d_aaa_lora) { tm.a1.weight = data; return true; } }
    else if (param_name == "time_mix.a2.weight") { if (data.size() == (size_t)model.config.d_aaa_lora * C) { tm.a2.weight = data; return true; } }
    else if (param_name == "time_mix.g1.weight") { if (data.size() == (size_t)C * model.config.d_gate_lora) { tm.g1.weight = data; return true; } }
    else if (param_name == "time_mix.g2.weight") { if (data.size() == (size_t)model.config.d_gate_lora * C) { tm.g2.weight = data; return true; } }
    else if (param_name == "time_mix.ln_x.weight") { if (data.size() == (size_t)C) { tm.ln_x_weight = data; return true; } }
    else if (param_name == "time_mix.ln_x.bias") { if (data.size() == (size_t)C) { tm.ln_x_bias = data; return true; } }
    else if (param_name == "channel_mix.ln2.weight") { if (data.size() == (size_t)C) { cm.ln2.weight = data; return true; } }
    else if (param_name == "channel_mix.ln2.bias") { if (data.size() == (size_t)C) { cm.ln2.bias = data; return true; } }
    else if (param_name == "channel_mix.x_k") { if (data.size() == (size_t)C) { cm.x_k = data; return true; } }
    else if (param_name == "channel_mix.key.weight") { if (data.size() == (size_t)C * F) { cm.key.weight = data; return true; } }
    else if (param_name == "channel_mix.value.weight") { if (data.size() == (size_t)F * C) { cm.value.weight = data; return true; } }
    
    return false;
}

} // namespace petrwkv