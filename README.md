# 🐾 PetRWKV — 本地智能宠物（可进化）

> 一只住在你手机里的 AI 宠物，越聊越懂你，越用越聪明。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![RWKV-v7](https://img.shields.io/badge/RWKV-v7--0.4B-green.svg)](https://github.com/BlinkDL/RWKV-LM)
[![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20iOS%20%7C%20Desktop-orange.svg)]()

## ✨ 核心特性

- **🧠 RWKV-v7 0.4B 全模态模型** — 文本/图片/音频/视频，原生集成，无外挂模块
- **🌙 夜间自进化** — 充电/闲置时自动整理对话、构建数据集、微调模型
- **📚 RAG 记忆库** — 对话历史自动入库，长期记忆不丢失
- **🐣 宠物进化系统** — 从蛋→幼崽→少年→成年→传说，5 阶段可视化进化
- **🔒 完全本地** — 推理、训练、存储全部在设备端，零云端依赖
- **⚡ 多后端加速** — ncnn / QNN(NPU) / CPU / WebGPU，自动选择最优后端

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    PetRWKV Flutter App                       │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Pet UI  │  Chat    │  Memory  │ Evolution│  Settings       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                    Core Services Layer                       │
├──────────┬──────────┬──────────┬────────────────────────────┤
│ RWKV     │ RAG      │ Training │ Background                 │
│ Inference│ Memory   │ Pipeline │ Scheduler                  │
│ Engine   │ System   │ (LoRA)   │ (充电/闲置检测)            │
├──────────┴──────────┴──────────┴────────────────────────────┤
│              C++ Native Engine (rwkv-mobile fork)            │
├──────────┬──────────┬──────────┬────────────────────────────┤
│ RWKV-v7  │ Vision   │ Audio    │ Video                      │
│ Core     │ Encoder  │ Encoder  │ Encoder                    │
│ (0.4B)   │ (ViT-S)  │ (Whisper-│ (Frame Extract             │
│          │          │  tiny)   │  + Vision)                 │
├──────────┴──────────┴──────────┴────────────────────────────┤
│         Hardware Backends: ncnn | QNN | CPU | WebGPU        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
PetRWKV/
├── app/                        # Flutter 前端应用
│   ├── lib/
│   │   ├── pet/                # 宠物系统（状态/进化/个性/UI）
│   │   ├── inference/          # RWKV 推理引擎（Dart FFI 桥接）
│   │   ├── rag/                # RAG 记忆系统
│   │   ├── training/           # 设备端微调管线
│   │   ├── chat/               # 对话界面
│   │   └── services/           # 后台服务（充电/闲置检测）
│   └── android/                # Android 平台配置
├── engine/                     # C++ 推理引擎（基于 rwkv-mobile）
│   ├── src/
│   │   ├── rwkv_v7.cpp        # RWKV-v7 核心推理
│   │   ├── multimodal/         # 多模态编码器
│   │   └── training/           # 设备端训练（LoRA）
│   └── backends/               # 硬件加速后端
├── training/                   # Python 训练管线（基于 RWKV-LM-V7）
│   ├── src/                    # 模型定义/数据集/训练器
│   ├── config/                 # 0.4B 模型配置
│   └── scripts/                # 训练/导出脚本
├── rag/                        # RAG 系统（本地向量检索）
├── .github/workflows/          # CI/CD（GitHub Actions）
└── docs/                       # 设计文档
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/rainrainrainbow/PetRWKV.git
cd PetRWKV
```

### 2. 安装依赖

```bash
# Flutter 依赖
cd app && flutter pub get && cd ..

# C++ 引擎构建（Android）
cd engine && mkdir build && cd build
cmake .. -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
    -DANDROID_NDK=$HOME/android-ndk-r25c \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/android-ndk-r25c/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja
```

### 3. 下载预训练模型

```bash
# 下载 0.4B 基础模型（已包含多模态编码器）
wget -O app/assets/model/pet_rwkv_0.4b.pth \
    https://huggingface.co/PetRWKV/pet-rwkv-0.4b/resolve/main/model.pth
```

### 4. 运行

```bash
cd app
flutter run
```

## 🐣 宠物进化系统

| 阶段 | 名称 | 条件 | 能力 |
|------|------|------|------|
| 🥚 Lv.0 | 神秘蛋 | 初始 | 基础对话 |
| 🐣 Lv.1 | 幼崽 | 完成首次微调 | 记住偏好 |
| 🐥 Lv.2 | 少年 | 累计 7 天微调 | 图片理解 |
| 🦊 Lv.3 | 成年 | 累计 30 天微调 | 音频理解 |
| 🐉 Lv.4 | 传说 | 累计 90 天微调 | 视频理解 + 完整人格 |

## 🌙 夜间自进化流程

```
[充电/闲置检测] → [对话历史整理] → [重要性评分]
       ↓                                    ↓
[选择性入库 RAG] ← ← ← ← ← ← ← [构建微调数据集]
                                            ↓
                                    [LoRA 微调 RWKV-v7]
                                            ↓
                                    [模型热更新 → 宠物进化]
```

## 🔧 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Flutter 3.44+ / Dart |
| 推理引擎 | C++ (rwkv-mobile fork) |
| 模型 | RWKV-v7 0.4B (全模态) |
| 训练 | PyTorch + pytorch-lightning 1.9.5 |
| RAG | SQLite + 自研轻量向量索引 |
| CI/CD | GitHub Actions |
| 参考 | [RWKV_APP](https://github.com/RWKV-APP/RWKV_APP), [rwkv-mobile](https://github.com/MollySophia/rwkv-mobile), [RWKV-LM-V7](https://github.com/RWKV-Vibe/RWKV-LM-V7) |

## 📄 License

Apache License 2.0
