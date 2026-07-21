import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrainDataset, ValDataset

"""
AMP
    就是配合 --amp 的实际实现：
    torch.amp.autocast("cuda")
    torch.amp.GradScaler(...)
    让前向和部分反向用混合精度。
梯度裁剪
    对应 --grad-clip，在 loss.backward() 之后、optimizer.step() 之前执行：
    clip_grad_norm_(model.parameters(), grad_clip)
    防止梯度爆炸。
epoch 末完整验证
    以前是训练过程中抽几个验证 batch，可能不代表整个验证集。
    现在每个 epoch 结束后，会完整跑一遍 val_loader，得到更可信的 avg_val_loss，并用它保存 best_model.pth。这对选最优模型很重要。

"""
def loss(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    month_idx: torch.Tensor,
    ice_threshold: float = 0.15,
    ice_weight: float = 2.0,
    return_components: bool = False,
) -> torch.Tensor:
    del month_idx

    outputs = outputs.squeeze(1)
    targets = targets.squeeze(1)

    with torch.no_grad():
        ocean_mask = (inputs[:, -1] != 1).float()
        eps = 1e-7
        y = targets.clamp(0, 1)
        if ice_weight != 1.0:
            ice_mask = (y >= ice_threshold).float()
            pixel_weight = ocean_mask * (1.0 + (ice_weight - 1.0) * ice_mask)
        else:
            pixel_weight = ocean_mask
        weight_sum = pixel_weight.sum(dim=(-2, -1)).clamp(min=1e-8)
        extreme = (y <= ice_threshold) | (y >= 1.0 - ice_threshold)

    x = outputs.clamp(eps, 1 - eps)
    bce = -(y * torch.log(x) + (1 - y) * torch.log(1 - x))
    mse = (y - x) ** 2

    loss_pixel = torch.where(extreme, bce, mse)
    weighted_loss = loss_pixel * pixel_weight
    loss_sample = weighted_loss.sum(dim=(-2, -1)) / weight_sum
    total_loss = loss_sample.mean()

    var_pred = x.var(dim=(-2, -1))
    var_true = y.var(dim=(-2, -1))
    var_loss = ((var_pred - var_true) ** 2).mean()
    total_loss = total_loss + 0.1 * var_loss

    if return_components:
        with torch.no_grad():
            mse_comp = (mse * (~extreme).float() * pixel_weight).sum(dim=(-2, -1)) / weight_sum
            bce_comp = (bce * extreme.float() * pixel_weight).sum(dim=(-2, -1)) / weight_sum
        return total_loss, mse_comp.mean(dim=0), bce_comp.mean(dim=0)

    return total_loss


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def evaluate_full_val(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> float:
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_inputs, val_targets, val_month_idx in val_loader:
            val_inputs = val_inputs.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)
            val_month_idx = val_month_idx.to(device, non_blocking=True)
            with _autocast(device, amp):
                val_outputs = model(val_inputs)
                val_loss = loss(val_inputs, val_outputs, val_targets, val_month_idx)
            val_losses.append(val_loss.item())
    return sum(val_losses) / len(val_losses) if val_losses else 0.0


def train_and_eval_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    tb_writer,
    output_dir: str = None,
    global_start_step: int = 0,
    grad_clip: float = 1.0,
    amp: bool = False,
    scaler: torch.amp.GradScaler = None,
    val_interval: int = 20,
):
    model.train()
    val_iter = iter(val_loader)
    train_losses = []
    val_loss = None

    for batch_idx, (inputs, targets, month_idx) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        month_idx = month_idx.to(device, non_blocking=True)

        model.train()
        with _autocast(device, amp):
            outputs = model(inputs)
            train_loss, train_mse, train_bce = loss(
                inputs,
                outputs,
                targets,
                month_idx,
                return_components=True,
            )

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(train_loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        train_loss_val = train_loss.item()
        train_losses.append(train_loss_val)

        current_step = global_start_step + batch_idx + 1
        lr = optimizer.param_groups[0]["lr"]
        tb_writer.add_scalar("lr", lr, current_step)
        tb_writer.add_scalar("loss/train", train_loss_val, current_step)
        for mon_idx in range(6):
            tb_writer.add_scalar(f"mse_loss/mon{mon_idx + 1}", train_mse[mon_idx].item(), current_step)
            tb_writer.add_scalar(f"bce_loss/mon{mon_idx + 1}", train_bce[mon_idx].item(), current_step)

        if val_interval > 0 and (batch_idx + 1) % val_interval == 0:
            model.eval()
            try:
                val_inputs, val_targets, val_month_idx = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_inputs, val_targets, val_month_idx = next(val_iter)

            val_inputs = val_inputs.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)
            val_month_idx = val_month_idx.to(device, non_blocking=True)

            with torch.no_grad():
                with _autocast(device, amp):
                    val_outputs = model(val_inputs)
                    val_loss = loss(val_inputs, val_outputs, val_targets, val_month_idx)
                tb_writer.add_scalar("loss/val", val_loss.item(), current_step)

        if val_loss is not None:
            print(
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Train Loss: {train_loss_val:.6f} | "
                f"Val Loss: {val_loss.item():.6f} | "
                f"LR: {lr:.2e}",
                flush=True,
            )
        else:
            print(
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Train Loss: {train_loss_val:.6f} | "
                f"LR: {lr:.2e}",
                flush=True,
            )

        if output_dir and current_step % 500 == 0:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{current_step}.pth")
            torch.save(
                {
                    "step": current_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss.item() if val_loss is not None else None,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
    avg_val_loss = evaluate_full_val(model, val_loader, device, amp=amp)
    tb_writer.add_scalar("loss/val_epoch", avg_val_loss, global_start_step + len(train_loader))

    return avg_train_loss, avg_val_loss, global_start_step + len(train_loader)


def get_dataloader(
    data_dir: str,
    batch_size: int = 8,
    shuffle: bool = False,
):
    dataset = TrainDataset(numpy_dir=Path(data_dir))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def get_val_dataloader(
    val_data_dir: str,
    batch_size: int = 8,
    shuffle: bool = False,
):
    dataset = ValDataset(val_numpy_path=Path(val_data_dir))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
