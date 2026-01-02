#!/usr/bin/env python3
"""
CosyVoice Flow LoRA 微调训练脚本

使用方法:
    python train.py              # 从头训练或自动恢复
    python train.py --resume     # 强制从 last.ckpt 恢复
    python train.py --fresh      # 强制从头开始训练

依赖:
    pip install pytorch-lightning einops

训练完成后:
    1. LoRA 权重保存到: output/flow_lora/flow_lora.pt
    2. 合并后的完整权重保存到: output/flow_lora/flow_merged.pt
    3. 使用 inference.py 进行推理测试
"""

import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

# 添加代码路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PRETRAINED_MODEL_DIR, DATA_DIR, OUTPUT_DIR,
    TRAIN_CONFIG, LORA_CONFIG
)

# 尝试导入无 prompt 训练配置
try:
    from config import NO_PROMPT_TRAINING_CONFIG
except ImportError:
    NO_PROMPT_TRAINING_CONFIG = {'enabled': False}

from flow_model import build_flow_model
from dataset import FlowFinetuneDataset, collate_fn


# ============================================================
# 训练终止条件配置
# ============================================================

STOP_CONFIG = {
    # Loss 低于此阈值时自动停止训练
    'loss_threshold': 0.35,

    # 连续多少个 epoch 没有改善就停止（EarlyStopping patience）
    'patience': 10,

    # 最小改善量，小于此值视为没有改善
    'min_delta': 0.005,
}


# ============================================================
# GPU 内存管理工具
# ============================================================

def aggressive_cleanup():
    """积极的内存清理"""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def log_gpu_memory(prefix=""):
    """记录 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix}[GPU] 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB, 峰值: {max_allocated:.2f}GB")


# ============================================================
# Lightning Module
# ============================================================

class FlowLightningModule(pl.LightningModule):
    def __init__(self, config, lora_config):
        super().__init__()
        self.save_hyperparameters()

        # 构建模型
        self.model = build_flow_model(pretrained_path=PRETRAINED_MODEL_DIR, device='cpu')

        # 设置 LoRA 微调
        if lora_config['use_lora']:
            self._setup_lora(lora_config)

        self.config = config
        self.lora_config = lora_config

    def _setup_lora(self, lora_config):
        """设置 LoRA 微调"""
        from lora import apply_lora_to_model

        stats = apply_lora_to_model(
            self.model,
            r=lora_config['lora_r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
        )

        print(f"\n{'='*50}")
        print(f"LoRA 配置:")
        print(f"  秩 (r): {lora_config['lora_r']}")
        print(f"  Alpha: {lora_config['lora_alpha']}")
        print(f"  Dropout: {lora_config['lora_dropout']}")
        print(f"  目标模块: {lora_config['target_modules']}")
        print(f"\nLoRA 统计:")
        print(f"  替换层数: {stats['replaced_layers']}")
        print(f"  原始参数: {stats['original_params']:,}")
        print(f"  LoRA 参数: {stats['lora_params']:,}")
        print(f"  可训练参数: {stats['trainable_params']:,}")
        print(f"  可训练比例: {stats['trainable_ratio']:.2f}%")
        print(f"{'='*50}\n")

    def forward(self, batch):
        return self.model.forward(batch, self.device)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        output = self.forward(batch)
        loss = output['loss']

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 每 20 个 batch 清理一次缓存（更频繁）
        if batch_idx % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个 batch 结束后的清理"""
        # 删除不需要的引用
        if batch is not None:
            del batch
        if outputs is not None and isinstance(outputs, dict):
            outputs.clear()

    def on_train_epoch_end(self):
        """每个 epoch 结束后的清理"""
        aggressive_cleanup()
        log_gpu_memory(f"\n[Epoch {self.current_epoch}] ")

    def configure_optimizers(self):
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.config['weight_decay'],
        )

        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config['warmup_steps']
        )

        estimated_total_steps = self.config['max_epochs'] * 100
        cosine_steps = max(1, estimated_total_steps - self.config['warmup_steps'])

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.config['min_learning_rate']
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config['warmup_steps']]
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]


# ============================================================
# 数据模块
# ============================================================

class FlowDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, max_feat_len, augmentation=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_feat_len = max_feat_len
        self.augmentation = augmentation

    def setup(self, stage=None):
        class TruncatedDataset(FlowFinetuneDataset):
            def __init__(self, data_dir, max_feat_len, augmentation=True):
                super().__init__(data_dir, augmentation=augmentation)
                self.max_feat_len = max_feat_len

            def __getitem__(self, idx):
                item = super().__getitem__(idx)
                if item is None:
                    return None

                feat_len = item['speech_feat'].shape[0]
                if feat_len > self.max_feat_len:
                    ratio = item['speech_token'].shape[0] / feat_len
                    max_token_len = int(self.max_feat_len * ratio)
                    item['speech_token'] = item['speech_token'][:max_token_len]
                    item['speech_feat'] = item['speech_feat'][:self.max_feat_len]

                return item

        self.train_dataset = TruncatedDataset(self.data_dir, self.max_feat_len, self.augmentation)
        print(f"数据集样本数: {len(self.train_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )


# ============================================================
# 自定义回调
# ============================================================

class GPUMemoryCallback(Callback):
    """积极的 GPU 内存管理回调"""

    def __init__(self, cleanup_every_n_batches=15):
        super().__init__()
        self.cleanup_every_n_batches = cleanup_every_n_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 更频繁地清理缓存
        if batch_idx % self.cleanup_every_n_batches == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def on_train_epoch_start(self, trainer, pl_module):
        """每个 epoch 开始前清理"""
        aggressive_cleanup()

    def on_train_epoch_end(self, trainer, pl_module):
        """每个 epoch 结束后清理"""
        aggressive_cleanup()
        log_gpu_memory(f"\n[Epoch {trainer.current_epoch}] ")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """保存 checkpoint 时清理"""
        aggressive_cleanup()

    def on_validation_end(self, trainer, pl_module):
        """验证结束后清理"""
        aggressive_cleanup()


class LossThresholdCallback(Callback):
    """当 Loss 低于阈值时自动停止训练"""

    def __init__(self, threshold=0.35):
        super().__init__()
        self.threshold = threshold
        self.triggered = False

    def on_train_epoch_end(self, trainer, pl_module):
        # 获取当前 epoch 的平均 loss
        current_loss = trainer.callback_metrics.get('train_loss_epoch')

        if current_loss is not None:
            loss_value = current_loss.item() if torch.is_tensor(current_loss) else current_loss

            if loss_value < self.threshold:
                print(f"\n{'='*50}")
                print(f"[自动停止] Loss ({loss_value:.4f}) 已低于阈值 ({self.threshold})")
                print(f"训练目标已达成，自动结束训练")
                print(f"{'='*50}\n")
                self.triggered = True
                trainer.should_stop = True


class ProgressReportCallback(Callback):
    """训练进度报告回调"""

    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics.get('train_loss_epoch')

        if current_loss is not None:
            loss_value = current_loss.item() if torch.is_tensor(current_loss) else current_loss

            if loss_value < self.best_loss - STOP_CONFIG['min_delta']:
                improvement = self.best_loss - loss_value
                self.best_loss = loss_value
                self.epochs_without_improvement = 0
                print(f"\n[进度] Epoch {trainer.current_epoch}: Loss={loss_value:.4f} (↓ {improvement:.4f}) ★ 新最佳")
            else:
                self.epochs_without_improvement += 1
                print(f"\n[进度] Epoch {trainer.current_epoch}: Loss={loss_value:.4f} (最佳: {self.best_loss:.4f}, 无改善: {self.epochs_without_improvement}轮)")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='CosyVoice Flow LoRA Training')
    parser.add_argument('--resume', action='store_true', help='从 last.ckpt 恢复训练')
    parser.add_argument('--fresh', action='store_true', help='强制从头开始训练')
    parser.add_argument('--ckpt', type=str, default=None, help='指定 checkpoint 路径')
    args = parser.parse_args()

    print("=" * 60)
    print("CosyVoice Flow LoRA 微调训练")
    print("=" * 60)

    # 显示训练模式
    if NO_PROMPT_TRAINING_CONFIG.get('enabled', False):
        print(f"\n[训练模式] 无 Prompt 训练模式")
        print(f"  - 模式: {NO_PROMPT_TRAINING_CONFIG.get('mode', 'full')}")
        if NO_PROMPT_TRAINING_CONFIG.get('mode') == 'mixed':
            print(f"  - 无 prompt 比例: {NO_PROMPT_TRAINING_CONFIG.get('no_prompt_ratio', 0.8) * 100:.0f}%")
        print(f"  - 推理时无需参考音频，完全依赖 LoRA 学到的音色")
    else:
        print(f"\n[训练模式] 标准模式（带 Prompt）")
        print(f"  - 使用语义泄漏防护策略")
        print(f"  - 推理时需要提供参考音频")

    # 显示终止条件
    print(f"\n终止条件:")
    print(f"  - Loss 阈值: {STOP_CONFIG['loss_threshold']} (低于此值自动停止)")
    print(f"  - EarlyStopping: {STOP_CONFIG['patience']} 轮无改善则停止")
    print(f"  - 最小改善量: {STOP_CONFIG['min_delta']}")

    # 启动前清理 GPU
    aggressive_cleanup()
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 总显存: {total_mem:.1f} GB")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 确定 checkpoint 路径
    ckpt_path = None
    if args.fresh:
        print("\n[模式] 强制从头开始训练")
    elif args.ckpt:
        if os.path.exists(args.ckpt):
            ckpt_path = args.ckpt
            print(f"\n[模式] 从指定 checkpoint 恢复: {ckpt_path}")
    elif args.resume:
        last_ckpt = os.path.join(OUTPUT_DIR, 'last.ckpt')
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"\n[模式] 从 last.ckpt 恢复训练")
    else:
        last_ckpt = os.path.join(OUTPUT_DIR, 'last.ckpt')
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"\n[模式] 自动检测到 last.ckpt，恢复训练")

    # 创建数据模块
    data_module = FlowDataModule(
        data_dir=DATA_DIR,
        batch_size=TRAIN_CONFIG['batch_size'],
        max_feat_len=TRAIN_CONFIG['max_feat_len'],
        augmentation=TRAIN_CONFIG['augmentation'],
    )

    # 创建模型
    aggressive_cleanup()
    model = FlowLightningModule(TRAIN_CONFIG, LORA_CONFIG)

    # 回调函数
    callbacks = [
        # GPU 内存管理（更频繁清理）
        GPUMemoryCallback(cleanup_every_n_batches=15),

        # Loss 阈值自动停止
        LossThresholdCallback(threshold=STOP_CONFIG['loss_threshold']),

        # 进度报告
        ProgressReportCallback(),

        # 保存最佳模型
        ModelCheckpoint(
            dirpath=OUTPUT_DIR,
            filename='flow_best_{epoch}_{train_loss_epoch:.4f}',
            monitor='train_loss_epoch',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
        ),

        # 早停（更敏感的设置）
        EarlyStopping(
            monitor='train_loss_epoch',
            patience=STOP_CONFIG['patience'],
            mode='min',
            min_delta=STOP_CONFIG['min_delta'],
            verbose=True,
        ),

        # 学习率监控
        LearningRateMonitor(logging_interval='step'),
    ]

    # Logger
    logger = TensorBoardLogger(save_dir=OUTPUT_DIR, name='logs')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=TRAIN_CONFIG['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=TRAIN_CONFIG['precision'],
        accumulate_grad_batches=TRAIN_CONFIG['accumulate_grad_batches'],
        gradient_clip_val=TRAIN_CONFIG['gradient_clip_val'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # 训练前测试
    print("\n测试初始 Loss...")
    aggressive_cleanup()
    data_module.setup()
    test_batch = next(iter(data_module.train_dataloader()))
    if test_batch is not None:
        model.eval()
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.model.to(device)
            output = model.model.forward(test_batch, device)
            print(f"初始 Loss: {output['loss'].item():.4f}")
            model.model.to('cpu')
            aggressive_cleanup()
        model.train()

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    try:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 60)
            print("[错误] GPU 内存不足！")
            print("建议:")
            print("  1. 减小 batch_size (当前: {})".format(TRAIN_CONFIG['batch_size']))
            print("  2. 减小 max_feat_len (当前: {})".format(TRAIN_CONFIG['max_feat_len']))
            print("  3. 关闭其他占用 GPU 的程序")
            print("=" * 60)
            aggressive_cleanup()
        raise

    # 保存最终模型
    if LORA_CONFIG['use_lora']:
        from lora import save_lora_weights, get_merged_state_dict

        aggressive_cleanup()

        # 保存 LoRA 权重（小文件）
        lora_path = os.path.join(OUTPUT_DIR, 'flow_lora.pt')
        save_lora_weights(model.model, lora_path)
        print(f"\nLoRA 权重保存至: {lora_path}")

        # 合并 LoRA 到原始权重（可被 CosyVoice 直接加载）
        merged_state_dict = get_merged_state_dict(model.model)
        merged_path = os.path.join(OUTPUT_DIR, 'flow_merged.pt')
        torch.save(merged_state_dict, merged_path)
        print(f"合并后完整模型保存至: {merged_path}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"TensorBoard 日志: {OUTPUT_DIR}/logs")
    print(f"查看: tensorboard --logdir {OUTPUT_DIR}/logs")
    print(f"\n下一步: python quick_inference.py --weight {OUTPUT_DIR}/flow_merged.pt --text \"测试文本\"")


if __name__ == '__main__':
    main()
