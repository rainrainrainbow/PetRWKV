#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace petrwkv {

constexpr int TOKEN_BOS = 0;
constexpr int TOKEN_EOS = 1;
constexpr int TOKEN_UNK = 2;
constexpr int TOKEN_PAD = 3;

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;
    
    bool load(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text, bool add_bos = true, int max_length = 0);
    std::string decode(const std::vector<int>& tokens);
    std::string decode_token(int token_id);
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    bool is_loaded() const { return !id_to_token_.empty(); }
    std::string token_to_bytes(int token_id) const;
    void build_trie();
    std::vector<int> encode_fast(const std::string& text);

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<int> byte_to_token_;
    std::vector<int> token_to_byte_;
    
    struct TrieNode {
        std::unordered_map<int, std::unique_ptr<TrieNode>> children;
        int token_id = -1;
    };
    std::unique_ptr<TrieNode> root_;
    
    void init_byte_encoding();
    std::vector<int> bpe_merge(const std::vector<int>& ids);
};

} // namespace petrwkv