#include "tokenizer.h"
#include <fstream>
#include <iostream>

namespace petrwkv {

void Tokenizer::init_byte_encoding() {
    byte_to_token_.resize(256, TOKEN_UNK);
    token_to_byte_.resize(id_to_token_.size(), -1);
    for (int i = 0; i < 256 && (i + 256) < (int)id_to_token_.size(); i++) {
        byte_to_token_[i] = i + 256;
        token_to_byte_[i + 256] = i;
    }
}

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream fs(vocab_path);
    if (!fs.is_open()) {
        std::cerr << "[Tokenizer] 无法打开词表文件: " << vocab_path << std::endl;
        return false;
    }
    
    token_to_id_.clear();
    id_to_token_.clear();
    
    std::string line;
    while (std::getline(fs, line)) {
        if (line.empty()) continue;
        size_t space_pos = line.rfind(' ');
        if (space_pos == std::string::npos) continue;
        
        std::string token_text = line.substr(0, space_pos);
        int token_id = std::stoi(line.substr(space_pos + 1));
        
        token_to_id_[token_text] = token_id;
        if (token_id >= (int)id_to_token_.size()) id_to_token_.resize(token_id + 1);
        id_to_token_[token_id] = token_text;
    }
    
    std::cout << "[Tokenizer] 加载了 " << id_to_token_.size() << " 个 token" << std::endl;
    init_byte_encoding();
    build_trie();
    return true;
}

void Tokenizer::build_trie() {
    root_ = std::make_unique<TrieNode>();
    for (const auto& [token_text, token_id] : token_to_id_) {
        auto* node = root_.get();
        for (char c : token_text) {
            int byte_val = static_cast<unsigned char>(c);
            if (!node->children[byte_val]) node->children[byte_val] = std::make_unique<TrieNode>();
            node = node->children[byte_val].get();
        }
        node->token_id = token_id;
    }
}

std::vector<int> Tokenizer::encode_fast(const std::string& text) {
    std::vector<int> result;
    size_t i = 0;
    while (i < text.size()) {
        auto* node = root_.get();
        size_t j = i;
        int last_found_id = -1;
        size_t last_found_pos = i;
        
        while (j < text.size() && node->children.count(static_cast<unsigned char>(text[j]))) {
            node = node->children[static_cast<unsigned char>(text[j])].get();
            j++;
            if (node->token_id >= 0) { last_found_id = node->token_id; last_found_pos = j; }
        }
        
        if (last_found_id >= 0) {
            result.push_back(last_found_id);
            i = last_found_pos;
        } else {
            int byte_val = static_cast<unsigned char>(text[i]);
            result.push_back(byte_val < (int)byte_to_token_.size() ? byte_to_token_[byte_val] : TOKEN_UNK);
            i++;
        }
    }
    return result;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, int max_length) {
    std::vector<int> tokens = encode_fast(text);
    if (add_bos) tokens.insert(tokens.begin(), TOKEN_BOS);
    if (max_length > 0 && (int)tokens.size() > max_length) tokens.resize(max_length);
    return tokens;
}

std::string Tokenizer::decode_token(int token_id) {
    if (token_id >= 0 && token_id < (int)id_to_token_.size()) return id_to_token_[token_id];
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;
    for (int token_id : tokens) {
        if (token_id == TOKEN_EOS || token_id == TOKEN_PAD) continue;
        result += decode_token(token_id);
    }
    return result;
}

std::string Tokenizer::token_to_bytes(int token_id) const {
    if (token_id >= 0 && token_id < (int)token_to_byte_.size() && token_to_byte_[token_id] >= 0)
        return std::string(1, static_cast<char>(token_to_byte_[token_id]));
    if (token_id >= 0 && token_id < (int)id_to_token_.size())
        return id_to_token_[token_id];
    return "";
}

std::vector<int> Tokenizer::bpe_merge(const std::vector<int>& ids) {
    return ids;
}

} // namespace petrwkv