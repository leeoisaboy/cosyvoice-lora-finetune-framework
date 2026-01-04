#!/usr/bin/env python3
"""
LLM + Flow 联合训练脚本

特点：
1. 同时微调 LLM 和 Flow，学习音色 + 吟诵风格
2. 无 Prompt 训练模式，推理时不需要参考音频
3. 使用 LoRA 减少训练参数和显存占用

用法:
    # 联合训练（推荐）
    python train_joint.py

    # 只训练 LLM（学习吟诵风格）
    python train_joint.py --mode llm_only

    # 只训练 Flow（学习音色，等价于原来的 train.py）
    python train_joint.py --mode flow_only

    # 从 checkpoint 恢复
    python train_joint.py --resume output/joint_last.ckpt
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
COSYVOICE_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, COSYVOICE_ROOT)

from config import (
    PRETRAINED_MODEL_DIR, DATA_DIR, OUTPUT_DIR,
    TRAIN_CONFIG, JOINT_TRAINING_CONFIG
)
from dataset import FlowFinetuneDataset, collate_fn


def clear_gpu_cache():
    """清空 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LossThresholdCallback(Callback):
    """当 loss 达到阈值时停止训练"""

    def __init__(
        self,
        llm_loss_threshold: Optional[float] = 2.0,
        flow_loss_threshold: Optional[float] = 0.3,
        train_loss_threshold: Optional[float] = None,
        check_on_epoch_end: bool = True,
    ):
        super().__init__()
        self.llm_loss_threshold = llm_loss_threshold
        self.flow_loss_threshold = flow_loss_threshold
        self.train_loss_threshold = train_loss_threshold
        self.check_on_epoch_end = check_on_epoch_end

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        if not self.check_on_epoch_end:
            return

        metrics = trainer.callback_metrics

        # 检查 LLM loss 阈值
        llm_loss = metrics.get('llm_loss_epoch')
        if llm_loss is not None and self.llm_loss_threshold is not None:
            if llm_loss <= self.llm_loss_threshold:
                print(f"\n[LLM] loss ({llm_loss:.4f}) 达到阈值 ({self.llm_loss_threshold})，停止训练")
                trainer.should_stop = True
                return

        # 检查 Flow loss 阈值
        flow_loss = metrics.get('flow_loss_epoch')
        if flow_loss is not None and self.flow_loss_threshold is not None:
            if flow_loss <= self.flow_loss_threshold:
                print(f"\n[Flow] loss ({flow_loss:.4f}) 达到阈值 ({self.flow_loss_threshold})，停止训练")
                trainer.should_stop = True
                return

        # 检查总 loss 阈值
        train_loss = metrics.get('train_loss_epoch')
        if train_loss is not None and self.train_loss_threshold is not None:
            if train_loss <= self.train_loss_threshold:
                print(f"\n[Total] loss ({train_loss:.4f}) 达到阈值 ({self.train_loss_threshold})，停止训练")
                trainer.should_stop = True
                return


class JointLightningModule(pl.LightningModule):
    """联合训练 Lightning 模块"""

    def __init__(
        self,
        training_mode: str = 'joint',
        learning_rate: float = 5e-5,
        min_lr: float = 1e-6,
        warmup_steps: int = 200,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.training_mode = training_mode
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # 延迟加载模型（在 setup 中加载）
        self.model = None

    def setup(self, stage: Optional[str] = None):
        """在训练开始时加载模型"""
        if self.model is None:
            from llm_flow_model import build_joint_model

            print(f"\n{'='*60}")
            print(f"构建联合训练模型 (模式: {self.training_mode})")
            print(f"{'='*60}")

            self.model = build_joint_model(
                pretrained_path=PRETRAINED_MODEL_DIR,
                device='cpu',  # 先在 CPU 上构建，Lightning 会自动移到 GPU
                training_mode=self.training_mode,
                llm_lora_config=JOINT_TRAINING_CONFIG.get('llm_lora'),
                flow_lora_config=JOINT_TRAINING_CONFIG.get('flow_lora'),
            )

    def forward(self, batch):
        return self.model(batch, self.device)

    def training_step(self, batch, batch_idx):
        losses = self(batch)

        # 记录 loss（on_step=True 显示瞬时值，on_epoch=True 计算平均值）
        self.log('train_loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if 'llm_loss' in losses:
            self.log('llm_loss', losses['llm_loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if 'flow_loss' in losses:
            self.log('flow_loss', losses['flow_loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if 'llm_acc' in losses:
            self.log('llm_acc', losses['llm_acc'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # 定期清理显存
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return losses['loss']

    def on_train_epoch_end(self):
        """每个 epoch 结束时打印平均 loss"""
        metrics = self.trainer.callback_metrics

        epoch = self.current_epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} 训练完成 - 平均指标:")
        print(f"{'='*60}")

        # 获取 epoch 平均值（Lightning 自动计算的 _epoch 后缀）
        avg_loss = metrics.get('train_loss_epoch')
        if avg_loss is not None:
            print(f"  平均 train_loss: {avg_loss:.4f}")

        avg_llm_loss = metrics.get('llm_loss_epoch')
        if avg_llm_loss is not None:
            print(f"  平均 llm_loss:   {avg_llm_loss:.4f}")

        avg_flow_loss = metrics.get('flow_loss_epoch')
        if avg_flow_loss is not None:
            print(f"  平均 flow_loss:  {avg_flow_loss:.4f}")

        avg_llm_acc = metrics.get('llm_acc_epoch')
        if avg_llm_acc is not None:
            print(f"  平均 llm_acc:    {avg_llm_acc:.4f}")

        print(f"{'='*60}\n")

        # 清理显存
        clear_gpu_cache()

    def configure_optimizers(self):
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"\n可训练参数数量: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warmup + Cosine 学习率调度
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                return max(self.min_lr / self.learning_rate, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


def main():
    parser = argparse.ArgumentParser(description='LLM + Flow 联合训练')
    parser.add_argument('--mode', type=str, default='joint',
                        choices=['joint', 'llm_only', 'flow_only'],
                        help='训练模式')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练 epochs（覆盖配置）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size（覆盖配置）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置）')
    args = parser.parse_args()

    # 清理显存
    clear_gpu_cache()

    print("=" * 60)
    print("CosyVoice LLM + Flow 联合训练")
    print("=" * 60)
    print(f"训练模式: {args.mode}")
    print(f"预训练模型: {PRETRAINED_MODEL_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 获取训练参数
    joint_config = JOINT_TRAINING_CONFIG
    train_config = TRAIN_CONFIG

    epochs = args.epochs or joint_config.get('max_epochs', 50)
    batch_size = args.batch_size or joint_config.get('batch_size', 1)
    lr = args.lr or joint_config.get('learning_rate', 5e-5)
    accumulate = joint_config.get('accumulate_grad_batches', 8)

    print(f"\n训练参数:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Accumulate: {accumulate}")
    print(f"  有效 batch: {batch_size * accumulate}")
    print(f"  学习率: {lr}")

    # 打印停止条件
    print(f"\n停止条件:")
    print(f"  早停: 10 个 epoch 无改善")
    print(f"  LLM loss 阈值: 1.5（防止过拟合）")
    print(f"  Flow loss 阈值: 0.3")

    # 检查数据
    if not os.path.exists(DATA_DIR):
        print(f"\n错误：数据目录不存在: {DATA_DIR}")
        print("请先运行 prepare_data.py 准备训练数据")
        return

    # 创建数据集
    print("\n加载数据集...")
    dataset = FlowFinetuneDataset(
        data_dir=DATA_DIR,
    )
    print(f"  样本数量: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 建议设为 0
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # 创建模型
    model = JointLightningModule(
        training_mode=args.mode,
        learning_rate=lr,
        min_lr=train_config.get('min_learning_rate', 1e-6),
        warmup_steps=train_config.get('warmup_steps', 200),
        weight_decay=train_config.get('weight_decay', 0.01),
    )

    # 创建 callbacks
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename=f'joint_{args.mode}' + '_{epoch:02d}_{train_loss:.4f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f'joint_{args.mode}_last'

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 早停机制：当 loss 不再下降时停止（增加耐心值以充分训练）
    early_stop_callback = EarlyStopping(
        monitor='train_loss_epoch',
        min_delta=0.001,        # 最小变化量
        patience=10,            # 容忍 10 个 epoch 不改善（从 5 增加到 10）
        verbose=True,
        mode='min',
    )

    # 阈值停止：防止 LLM 过拟合
    # LLM loss 过低（<1.5）会导致过拟合，出现"自说自话"现象
    # 最佳 LLM loss 范围：1.5 ~ 2.5
    threshold_callback = LossThresholdCallback(
        llm_loss_threshold=1.5,     # LLM loss 达到 1.5 停止，防止过拟合
        flow_loss_threshold=0.3,    # Flow loss 可以继续降低
        train_loss_threshold=None,  # 不限制总 loss
    )

    logger = TensorBoardLogger(
        save_dir=OUTPUT_DIR,
        name='joint_logs',
        version=args.mode,
    )

    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=train_config.get('precision', '16-mixed'),
        accumulate_grad_batches=accumulate,
        gradient_clip_val=train_config.get('gradient_clip_val', 1.0),
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback, threshold_callback],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.fit(
        model,
        train_dataloaders=dataloader,
        ckpt_path=args.resume,
    )

    # 保存合并后的权重
    print("\n保存合并后的权重...")
    from llm_flow_model import get_joint_merged_state_dict

    merged = get_joint_merged_state_dict(model.model)

    if 'llm' in merged:
        llm_path = os.path.join(OUTPUT_DIR, f'llm_merged_{args.mode}.pt')
        torch.save(merged['llm'], llm_path)
        print(f"  LLM 权重: {llm_path}")

    if 'flow' in merged:
        flow_path = os.path.join(OUTPUT_DIR, f'flow_merged_{args.mode}.pt')
        torch.save(merged['flow'], flow_path)
        print(f"  Flow 权重: {flow_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n下一步：使用 inference_joint.py 进行无 prompt 推理")


if __name__ == '__main__':
    main()
