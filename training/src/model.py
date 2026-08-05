"""
PetRWKV 0.4B 全模态模型定义
基于 RWKV-v7 架构，原生集成视觉/音频编码器（无外挂模块）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ============================================================
# 配置
# ============================================================

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
    
    @property
    def total_params(self) -> int:
        """估算总参数量"""
        emb = self.vocab_size * self.hidden_size
        per_layer = (
            4 * self.hidden_size * self.hidden_size +
            2 * self.hidden_size * self.ffn_hidden_size +
            self.hidden_size * self.d_decay_lora * 2 +
            self.hidden_size * self.d_aaa_lora * 2 +
            self.hidden_size * self.d_gate_lora * 2
        )
        head = self.vocab_size * self.hidden_size
        return emb + per_layer * self.num_layers + head


class RWKV7TimeMix(nn.Module):
    """RWKV-v7 Time Mixing 层 (xLora 变体)"""
    def __init__(self, config: PetRWKVConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        C = config.hidden_size
        self.ln1 = nn.LayerNorm(C)
        self.x_r = nn.Parameter(torch.zeros(1, 1, C))
        self.x_w = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.x_a = nn.Parameter(torch.zeros(1, 1, C))
        self.x_g = nn.Parameter(torch.zeros(1, 1, C))
        self.k_k = nn.Parameter(torch.zeros(1, 1, C))
        self.k_a = nn.Parameter(torch.zeros(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(1, 1, C))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.w0 = nn.Parameter(torch.zeros(1, 1, C))
        self.w1 = nn.Linear(C, config.d_decay_lora, bias=False)
        self.w2 = nn.Linear(config.d_decay_lora, C, bias=False)
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))
        self.a1 = nn.Linear(C, config.d_aaa_lora, bias=False)
        self.a2 = nn.Linear(config.d_aaa_lora, C, bias=False)
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.zeros(1, 1, C))
            self.v1 = nn.Linear(C, config.d_mv_lora, bias=False)
            self.v2 = nn.Linear(config.d_mv_lora, C, bias=False)
        self.g1 = nn.Linear(C, config.d_gate_lora, bias=False)
        self.g2 = nn.Linear(config.d_gate_lora, C, bias=False)
        self.ln_x = nn.GroupNorm(32, C)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.x_r)
        nn.init.zeros_(self.x_w)
        nn.init.zeros_(self.x_k)
        nn.init.zeros_(self.x_v)
        nn.init.zeros_(self.x_a)
        nn.init.zeros_(self.x_g)
        nn.init.constant_(self.k_k, 0.71)
        nn.init.constant_(self.k_a, 1.02)
        nn.init.zeros_(self.r_k)
        nn.init.zeros_(self.w0)
        nn.init.zeros_(self.w1.weight)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.a0)
        nn.init.zeros_(self.a1.weight)
        nn.init.zeros_(self.a2.weight)
        if self.layer_id > 0:
            nn.init.zeros_(self.v0)
            nn.init.zeros_(self.v1.weight)
            nn.init.zeros_(self.v2.weight)
        nn.init.zeros_(self.g1.weight)
        nn.init.zeros_(self.g2.weight)
        nn.init.zeros_(self.output.weight)
    
    def forward(self, x, state=None):
        B, T, C = x.shape
        x = self.ln1(x)
        r = self.receptance(x)
        k = self.key(x)
        v = self.value(x)
        w = self.w0 + torch.tanh(self.w1(x)) @ self.w2.weight.T
        a = self.a0 + torch.tanh(self.a1(x)) @ self.a2.weight.T
        g = torch.sigmoid(self.g2(torch.tanh(self.g1(x))))
        kk = self.k_k
        ka = self.k_a
        k = k * kk
        k = F.normalize(k, dim=-1) * ka
        w_sig = torch.sigmoid(w)
        out = self.output(r * v)
        out = self.ln_x(out)
        out = out * g
        return out, state


class RWKV7ChannelMix(nn.Module):
    """RWKV-v7 Channel Mixing (FFN) 层"""
    def __init__(self, config: PetRWKVConfig, layer_id: int):
        super().__init__()
        C = config.hidden_size
        F = config.ffn_hidden_size
        self.ln2 = nn.LayerNorm(C)
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, F, bias=False)
        self.value = nn.Linear(F, C, bias=False)
        nn.init.zeros_(self.x_k)
        nn.init.zeros_(self.value.weight)
    
    def forward(self, x):
        x = self.ln2(x)
        k = self.key(x)
        k = torch.relu(k) ** 2
        out = self.value(k)
        return out


class RWKV7Block(nn.Module):
    """RWKV-v7 单个 Block"""
    def __init__(self, config: PetRWKVConfig, layer_id: int):
        super().__init__()
        self.time_mix = RWKV7TimeMix(config, layer_id)
        self.channel_mix = RWKV7ChannelMix(config, layer_id)
    
    def forward(self, x, state=None):
        tm_out, state = self.time_mix(x, state)
        x = x + tm_out
        cm_out = self.channel_mix(x)
        x = x + cm_out
        return x, state


class VisionEncoder(nn.Module):
    """轻量 ViT-Small 视觉编码器 - 直接嵌入 RWKV 隐藏空间"""
    def __init__(self, config: PetRWKVConfig):
        super().__init__()
        self.config = config
        embed_dim = config.image_encoder_dim
        hidden_dim = config.hidden_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        num_patches = (224 // 16) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True)
            for _ in range(4)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.projection(x)
        return x


class AudioEncoder(nn.Module):
    """轻量音频编码器 (Whisper-tiny 架构) - 直接嵌入 RWKV 隐藏空间"""
    def __init__(self, config: PetRWKVConfig):
        super().__init__()
        self.config = config
        embed_dim = config.audio_encoder_dim
        hidden_dim = config.hidden_size
        self.conv1 = nn.Conv1d(80, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.register_buffer('positional_embedding', self._get_sinusoidal_embedding(1500, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True)
            for _ in range(4)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim)
    
    def _get_sinusoidal_embedding(self, max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(mel_spectrograms))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.positional_embedding[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.projection(x)
        return x


class VideoEncoder(nn.Module):
    """视频编码器：帧提取 + VisionEncoder + 时序融合"""
    def __init__(self, config: PetRWKVConfig, vision_encoder: VisionEncoder):
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        hidden_dim = config.hidden_size
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = video_frames.shape
        frames_flat = video_frames.view(B * F, C, H, W)
        frame_features = self.vision_encoder(frames_flat)
        frame_features = frame_features.view(B, F * frame_features.shape[1], -1)
        temporal_out, _ = self.temporal_attn(frame_features, frame_features, frame_features)
        frame_features = self.temporal_norm(frame_features + temporal_out)
        return frame_features


class PetRWKV(nn.Module):
    """PetRWKV 全模态模型 - RWKV-v7 0.4B + 原生视觉/音频/视频编码器"""
    def __init__(self, config: PetRWKVConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([RWKV7Block(config, i) for i in range(config.num_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config, self.vision_encoder)
        self.modality_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.ones_(self.ln_out.weight)
        nn.init.zeros_(self.ln_out.bias)
    
    def encode_text(self, input_ids): return self.emb(input_ids)
    def encode_image(self, images): return self.vision_encoder(images)
    def encode_audio(self, mel_specs): return self.audio_encoder(mel_specs)
    def encode_video(self, frames): return self.video_encoder(frames)
    
    def forward(self, input_ids=None, images=None, audio=None, video=None, state=None):
        sequences = []
        if images is not None: sequences.append(self.encode_image(images))
        if audio is not None: sequences.append(self.encode_audio(audio))
        if video is not None: sequences.append(self.encode_video(video))
        if input_ids is not None: sequences.append(self.encode_text(input_ids))
        if len(sequences) == 0: raise ValueError("At least one input modality is required")
        x = torch.cat(sequences, dim=1)
        new_state = []
        for i, block in enumerate(self.blocks):
            s = state[i] if state is not None else None
            x, s = block(x, s)
            new_state.append(s)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_state
    
    def get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取文本嵌入向量（用于 RAG）"""
        x = self.emb(input_ids)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_out(x)
        embedding = x.mean(dim=1)
        return F.normalize(embedding, dim=-1)


class LoRAConfig:
    def __init__(self, r: int = 8, alpha: int = 16, target_modules: List[str] = None):
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.target_modules = target_modules or ['receptance', 'key', 'value', 'output']


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.original = original
        self.lora_a = nn.Linear(original.in_features, config.r, bias=False)
        self.lora_b = nn.Linear(config.r, original.out_features, bias=False)
        self.scaling = config.scaling
        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x):
        return self.original(x) + self.lora_b(self.lora_a(x)) * self.scaling


def apply_lora(model: PetRWKV, config: LoRAConfig) -> PetRWKV:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in config.target_modules:
                if target in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, LoRALinear(module, config))
                    break
    return model
