"""
PetRWKV 设备端 LoRA 微调训练器
基于 pytorch-lightning 1.9.5 (RWKV-LM-V7 要求版本)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from typing import Optional, Dict, Any
import json
import os

from .model import PetRWKV, PetRWKVConfig, LoRAConfig, apply_lora


class PetConversationDataset(Dataset):
    """宠物对话数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载 JSONL 数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建对话格式
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # 拼接为训练格式
        text = f"Instruction: {instruction}\nResponse: {output}"
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # 转换为 tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # 创建 attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # 创建 labels (与 input_ids 相同，用于因果语言建模)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class PetRWKVTrainer(pl.LightningModule):
    """PetRWKV 训练器"""
    
    def __init__(
        self,
        config: PetRWKVConfig,
        lora_config: Optional[LoRAConfig] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 初始化模型
        self.model = PetRWKV(config)
        
        # 应用 LoRA
        if lora_config is not None:
            self.model = apply_lora(self.model, lora_config)
            self.lora_config = lora_config
        else:
            self.lora_config = None
        
        # 冻结非 LoRA 参数
        if self.lora_config is not None:
            self._freeze_non_lora()
        
        # 保存超参数
        self.save_hyperparameters(ignore=['model'])
    
    def _freeze_non_lora(self):
        """冻结非 LoRA 参数"""
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    
    def forward(self, batch):
        input_ids = batch['input_ids']
        
        # 前向传播
        logits, _ = self.model(input_ids=input_ids)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['labels']
        
        # 计算交叉熵损失
        # 移动 logits 以匹配 labels 的序列长度
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # 记录指标
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['labels']
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # 计算困惑度
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_perplexity', perplexity, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # 只优化 LoRA 参数
        if self.lora_config is not None:
            lora_params = [
                p for n, p in self.model.named_parameters()
                if 'lora_' in n and p.requires_grad
            ]
            params = lora_params
        else:
            params = self.model.parameters()
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
    
    def save_pretrained(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))
        
        # 保存配置
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'ffn_hidden_size': self.config.ffn_hidden_size,
            'head_size': self.config.head_size,
            'd_decay_lora': self.config.d_decay_lora,
            'd_aaa_lora': self.config.d_aaa_lora,
            'd_mv_lora': self.config.d_mv_lora,
            'd_gate_lora': self.config.d_gate_lora,
            'image_encoder_dim': self.config.image_encoder_dim,
            'audio_encoder_dim': self.config.audio_encoder_dim,
            'ctx_len': self.config.ctx_len,
        }
        
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # 保存 LoRA 配置
        if self.lora_config is not None:
            lora_dict = {
                'r': self.lora_config.r,
                'alpha': self.lora_config.alpha,
                'target_modules': self.lora_config.target_modules,
            }
            with open(os.path.join(save_dir, 'lora_config.json'), 'w') as f:
                json.dump(lora_dict, f, indent=2)


def train_pet_model(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    config: Optional[PetRWKVConfig] = None,
    lora_config: Optional[LoRAConfig] = None,
    batch_size: int = 4,
    max_epochs: int = 10,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
):
    """
    训练宠物模型
    
    Args:
        train_data_path: 训练数据路径 (JSONL)
        val_data_path: 验证数据路径 (JSONL)
        output_dir: 输出目录
        config: 模型配置
        lora_config: LoRA 配置
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        learning_rate: 学习率
        num_workers: 数据加载线程数
    """
    # 默认配置
    if config is None:
        config = PetRWKVConfig()
    
    # 简单的 tokenizer (实际使用时需要加载真实的 tokenizer)
    class SimpleTokenizer:
        def encode(self, text):
            # 这里应该使用真实的 tokenizer
            # 暂时用字符级别的分词
            return [ord(c) % config.vocab_size for c in text]
    
    tokenizer = SimpleTokenizer()
    
    # 创建数据集
    train_dataset = PetConversationDataset(train_data_path, tokenizer, config.ctx_len)
    val_dataset = PetConversationDataset(val_data_path, tokenizer, config.ctx_len)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # 创建训练器
    model = PetRWKVTrainer(
        config=config,
        lora_config=lora_config,
        learning_rate=learning_rate,
    )
    
    # 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='pet-rwkv-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # PyTorch Lightning 训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',  # 混合精度训练
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=output_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 保存最终模型
    model.save_pretrained(os.path.join(output_dir, 'final'))
    
    print(f"Training completed! Model saved to {output_dir}")
    
    return model
