#!/usr/bin/env python3

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 显存优化设置（解决4090等新GPU显存碎片问题）
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.tensorboard import SummaryWriter

from model_geo_TTT_5 import model as create_model
from utils import train_and_eval_one_epoch, get_dataloader, get_val_dataloader


def main():
    parser = argparse.ArgumentParser(description='Swin Transformer Training')
    
    # 模型参数
    parser.add_argument('--in-chans', type=int, default=9,
                        help='输入通道数 (默认: 9)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练模型路径，加载后默认冻结其他层')
    parser.add_argument('--train-only', type=str, default=None,
                        help='指定只训练的层(逗号分隔)，如 "decoder" 或 "decoder.stage1"')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (默认: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率 (默认: 1e-4)')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='最小学习率 (默认: 1e-5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减 (默认: 0.05)')
    
    # 数据路径
    parser.add_argument('--train-data-dir', type=str,
                        default='./numpy/train',
                        help='训练数据目录')
    parser.add_argument('--val-data-dir', type=str,
                        default='./numpy/val',
                        help='验证数据目录 (会查找目录下的 val.npy)')
    
    # 其他设置
    parser.add_argument('--output-dir', type=str, 
                        default='./weights_TTT_5',
                        help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda 或 cpu)')
    parser.add_argument('--log-dir', type=str,
                        default='./runs',
                        help='TensorBoard 日志目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard 设置
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")
    
    # 创建数据加载器
    print(f"加载训练数据: {args.train_data_dir}")
    train_loader = get_dataloader(
        data_dir=args.train_data_dir,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    print(f"加载验证数据: {args.val_data_dir}")
    val_loader = get_val_dataloader(
        val_data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 打印数据集信息
    train_dataset = train_loader.dataset
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"每轮 batch 数: {len(train_loader)}")
    
    # 创建模型
    print(f"创建模型 (in_chans={args.in_chans})")
    model = create_model(num_classes=1, in_chans=args.in_chans).to(device)
    
    # 加载预训练权重并冻结指定层
    if args.pretrained:
        print(f"加载预训练权重: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model_dict = model.state_dict()
        # 过滤：只加载形状匹配的权重
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"加载了 {len(filtered_dict)} 层权重")
        
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻指定层
        if args.train_only:
            train_keys = [k.strip() for k in args.train_only.split(',')]
            frozen_count = 0
            trainable_count = 0
            for name, param in model.named_parameters():
                should_train = any(key in name for key in train_keys)
                if should_train:
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    frozen_count += 1
            print(f"冻结: {frozen_count} 层, 训练: {trainable_count} 层 (仅 {args.train_only})")
        else:
            print("已加载预训练权重，全部层已冻结")
    else:
        # 未使用预训练时，默认全部可训练
        pass
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    model_info = f"Total params: {total_params / 1e6:.2f}M, Trainable: {trainable_params / 1e6:.2f}M"
    tb_writer.add_text('model/info', model_info)
    
    # 优化器 (只优化可训练参数)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"优化器: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")

    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.lr_min / args.lr,  # 从 lr 衰减到 lr_min
        total_iters=args.epochs
    )
    print(f"学习率调度: LinearLR ({args.lr} -> {args.lr_min})")

    # 训练循环
    print("\n" + "="*60, flush=True)
    print("开始训练", flush=True)
    print("="*60, flush=True)

    best_val_loss = float('inf')
    global_step = 0  # 全局 step，用于 TensorBoard 连续记录

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)

        # 训练并验证（每个训练 batch 后立即验证）
        train_loss, val_loss, global_step = train_and_eval_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            tb_writer=tb_writer,
            output_dir=str(output_dir),
            global_start_step=global_step
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"保存最佳模型: {save_path} (val_loss={val_loss:.6f})", flush=True)

        # 更新学习率调度器
        scheduler.step()
    
    # 保存最终模型
    save_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, save_path)
    print(f"\n保存最终模型: {save_path}", flush=True)
    
    # 关闭 TensorBoard writer
    tb_writer.close()
    
    print("\n" + "="*60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("="*60)


if __name__ == '__main__':
    main()
