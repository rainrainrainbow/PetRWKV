"""
PetRWKV 0.4B 模型配置
基于 RWKV-v7 架构
"""

from dataclasses import dataclass


@dataclass
class PetRWKVConfig:
    """0.4B 全模态模型配置"""
    # RWKV 核心
    vocab_size: int = 65536
    hidden_size: int = 1024       # 0.4B: D=1024
    num_layers: int = 24          # L=24
    ffn_hidden_size: int = 4096   # 4x hidden
    head_size: int = 64           # v7 head dim
    
    # LoRA 维度 (0.4B 规格)
    d_decay_lora: int = 64
    d_aaa_lora: int = 64
    d_mv_lora: int = 32
    d_gate_lora: int = 128
    
    # 多模态
    image_encoder_dim: int = 384   # ViT-Small
    image_seq_len: int = 256       # 图像 token 数
    audio_encoder_dim: int = 384   # Whisper-tiny
    audio_seq_len: int = 1500      # 音频 token 数
    video_max_frames: int = 8      # 视频最大帧数
    
    # 训练
    ctx_len: int = 2048
    dropout: float = 0.1


# 预定义配置
PET_RWKV_0_4B = PetRWKVConfig()

PET_RWKV_0_4B_LORA = {
    'r': 8,
    'alpha': 16,
    'target_modules': ['receptance', 'key', 'value', 'output'],
}
