#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>

namespace petrwkv {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline float relu2(float x) { float r = x > 0 ? x : 0; return r * r; }

inline float silu(float x) { return x * sigmoid(x); }

inline float dot_product(const float* a, const float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

inline void vec_mul(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

inline void vec_add(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

inline void vec_scale(float* out, const float* x, float s, int n) {
    for (int i = 0; i < n; i++) out[i] = x[i] * s;
}

inline void vec_tanh(float* out, const float* x, int n) {
    for (int i = 0; i < n; i++) out[i] = std::tanh(x[i]);
}

inline void vec_sigmoid(float* out, const float* x, int n) {
    for (int i = 0; i < n; i++) out[i] = sigmoid(x[i]);
}

inline void vec_normalize(float* out, const float* x, int n) {
    float sum_sq = 0;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv_norm = 1.0f / (std::sqrt(sum_sq) + 1e-8f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_norm;
}

inline void matmul(float* C, const float* A, const float* B, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
    }
}

inline void matmul_bt(float* C, const float* A, const float* B_T, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            const float* a_row = &A[m * K];
            const float* b_col = &B_T[n * K];
            for (int k = 0; k < K; k++) sum += a_row[k] * b_col[k];
            C[m * N + n] = sum;
        }
    }
}

inline void layer_norm(float* out, const float* x, const float* weight, const float* bias, int n, float eps = 1e-5f) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv_std * weight[i] + (bias ? bias[i] : 0);
}

inline void group_norm(float* out, const float* x, const float* gamma, const float* beta, int n, int groups, float eps = 1e-5f) {
    int group_size = n / groups;
    for (int g = 0; g < groups; g++) {
        const float* x_g = x + g * group_size;
        float* out_g = out + g * group_size;
        float mean = 0, var = 0;
        for (int i = 0; i < group_size; i++) mean += x_g[i];
        mean /= group_size;
        for (int i = 0; i < group_size; i++) var += (x_g[i] - mean) * (x_g[i] - mean);
        var /= group_size;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < group_size; i++) out_g[i] = (x_g[i] - mean) * inv_std * gamma[g * group_size + i] + (beta ? beta[g * group_size + i] : 0);
    }
}

inline void softmax(float* out, const float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = std::exp(x[i] - max_val); sum += out[i]; }
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int i = 0; i < n; i++) out[i] *= inv_sum;
}

inline int sample_top_p(const float* logits, int n, float temp, float top_p, std::mt19937& rng) {
    std::vector<float> probs(n);
    if (temp > 0) { for (int i = 0; i < n; i++) probs[i] = logits[i] / temp; }
    softmax(probs.data(), probs.data(), n);
    std::vector<std::pair<float, int>> sorted(n);
    for (int i = 0; i < n; i++) sorted[i] = {probs[i], i};
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    float cumsum = 0; int cutoff = n;
    for (int i = 0; i < n; i++) { cumsum += sorted[i].first; if (cumsum > top_p) { cutoff = i + 1; break; } }
    std::uniform_real_distribution<float> dist(0, cumsum);
    float r = dist(rng); cumsum = 0;
    for (int i = 0; i < cutoff; i++) { cumsum += sorted[i].first; if (r <= cumsum) return sorted[i].second; }
    return sorted[0].second;
}

inline int sample_argmax(const float* logits, int n) {
    int idx = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[idx]) idx = i;
    return idx;
}

} // namespace petrwkv