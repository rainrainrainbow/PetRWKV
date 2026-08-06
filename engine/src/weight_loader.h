#pragma once
#include "rwkv_v7.h"
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

namespace petrwkv {

#pragma pack(push, 1)
struct WeightHeader {
    char magic[8];
    uint32_t format_version;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t ffn_hidden_size;
    uint32_t head_size;
    uint32_t ctx_len;
    uint32_t d_decay_lora;
    uint32_t d_aaa_lora;
    uint32_t d_mv_lora;
    uint32_t d_gate_lora;
    uint32_t weight_count;
    uint64_t total_size;
    uint32_t flags;
    uint32_t reserved[8];
};
#pragma pack(pop)

static_assert(sizeof(WeightHeader) == 128, "WeightHeader must be 128 bytes");

#pragma pack(push, 1)
struct TensorHeader {
    char name[64];
    uint32_t ndim;
    uint32_t dims[3];
    uint32_t dtype;
    uint64_t offset;
    uint64_t data_size;
};
#pragma pack(pop)

static_assert(sizeof(TensorHeader) == 96, "TensorHeader must be 96 bytes");

class WeightLoader {
public:
    static bool load_from_bin(RWKV7Model& model, const std::string& path);
    static bool load_from_ggml(RWKV7Model& model, const std::string& path);
    static bool load(RWKV7Model& model, const std::string& path);
    static bool save_to_bin(const RWKV7Model& model, const std::string& path);
    static std::string get_last_error() { return last_error_; }

private:
    static std::string last_error_;
    static bool read_tensor(std::ifstream& fs, std::vector<float>& tensor, const TensorHeader& hdr);
    static bool parse_tensor_name(const std::string& full_name, int& layer_idx, std::string& param_name);
    static bool assign_tensor(RWKV7Model& model, int layer_idx, const std::string& param_name, const std::vector<float>& data);
};

} // namespace petrwkv