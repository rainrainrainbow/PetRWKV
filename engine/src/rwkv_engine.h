#pragma once

#include <vector>
#include <string>

// RWKV-v7 推理引擎 C API
extern "C" {

// 加载模型
// @param model_path 模型文件路径
// @param backend 硬件后端 (可选: "ncnn", "qnn", "cpu", "webgpu")
// @return 0 成功, <0 失败
int load_model(const char* model_path, const char* backend);

// 卸载模型
void unload_model();

// 文本生成
// @param prompt 输入提示
// @param max_tokens 最大生成 token 数
// @param temperature 采样温度
// @param top_p Top-p 采样参数
// @return 生成的文本 (需要调用者释放内存)
char* generate(const char* prompt, int max_tokens, float temperature, float top_p);

// 图片理解
// @param image_path 图片文件路径
// @param prompt 文本提示
// @param max_tokens 最大生成 token 数
// @return 理解结果 (需要调用者释放内存)
char* understand_image(const char* image_path, const char* prompt, int max_tokens);

// 音频理解
// @param audio_path 音频文件路径
// @param prompt 文本提示
// @param max_tokens 最大生成 token 数
// @return 理解结果 (需要调用者释放内存)
char* understand_audio(const char* audio_path, const char* prompt, int max_tokens);

// 获取文本嵌入
// @param text 输入文本
// @return 嵌入向量 (逗号分隔的浮点数, 需要调用者释放内存)
char* get_embedding(const char* text);

// LoRA 微调
// @param dataset_path 训练数据集路径 (JSONL 格式)
// @param epochs 训练轮数
// @param learning_rate 学习率
// @return 0 成功, <0 失败
int train_lora(const char* dataset_path, int epochs, float learning_rate);

} // extern "C"
