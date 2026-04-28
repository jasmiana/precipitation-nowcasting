import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from dataset import get_dataloaders
from model import Seq2SeqConvLSTM
from improved_model import DeepSeq2SeqConvLSTM
import os
import cv2
import numpy as np
import time

EPOCHS = 20         # 50~100 Recommended
ADAM_LR = 5e-4
BATCH_SIZE = 16

def save_visualization(preds, targets, epoch, batch_idx, out_dir="train_outputs_improved"):
    os.makedirs(out_dir, exist_ok=True)
    # preds, targets: (B, T, 1, 64, 64) CPU tensors
    # 提取序列第一条数据
    p_seq = preds[0].detach().numpy() * 255.0  # (10, 1, 64, 64)
    t_seq = targets[0].detach().numpy() * 255.0
    
    # 拼成一行
    p_img = np.hstack([p_seq[i, 0] for i in range(10)])
    t_img = np.hstack([t_seq[i, 0] for i in range(10)])
    
    # 拼成两行
    combined = np.vstack((t_img, p_img))
    
    cv2.imwrite(f"{out_dir}/ep{epoch}_b{batch_idx}.png", combined)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # model = Seq2SeqConvLSTM(hidden_dim=64).to(device)
    model = DeepSeq2SeqConvLSTM().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.BCELoss()
    
    scaler = GradScaler()
    
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    accum_steps = 1
    
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # 计划采样衰减：前几个 epoch 是 1.0 (全 GroundTruth)，后面逐渐降到 0.0       
        # 修改前：
        # teacher_forcing_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.7)))

        # 修改后：前 5 个 epoch 保持 1.0，之后再缓慢下降
        warmup_epochs = 5
        # if epoch < warmup_epochs:
        #     teacher_forcing_ratio = 1.0
        # else:
        #     teacher_forcing_ratio = max(0.0, 1.0 - ((epoch - warmup_epochs) / (epochs * 0.7)))

        # 修改后：不要让它完全掉到 0，给它留至少 20% - 30% 的拐杖
        if epoch < warmup_epochs:
            teacher_forcing_ratio = 1.0
        else:
            # 衰减到最后，保留 0.3 的比例
            decayed = 1.0 - ((epoch - warmup_epochs) / (epochs * 0.7))
            teacher_forcing_ratio = max(0.3, decayed)

        optimizer.zero_grad()
        for batch_idx, (data_x, data_y) in enumerate(train_loader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            
            with autocast():
                output = model(data_x, future_steps=10, teacher_forcing_ratio=teacher_forcing_ratio, target=data_y)
                loss = criterion(output, data_y)
                # 因为加入了梯度累计，loss要除以累计步数
                # loss = loss / accum_steps
                
            scaler.scale(loss).backward()
            
            # Gradient Explosion?
            # ======== 新增：梯度裁剪 ========
            # 必须在 step 之前 unscale 梯度，否则裁剪的是缩放后的假梯度
            scaler.unscale_(optimizer) 
            # 裁剪阈值设为 1.0 到 5.0 之间通常比较合适
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
            # ===============================

            # if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
            train_loss += loss.item() * accum_steps # 恢复显示的标尺
            
            if batch_idx % 50 == 0:
                current = time.time()
                print(f"Epoch [{epoch}/{epochs}] \t Batch [{batch_idx}/{len(train_loader)}] \t Loss: {loss.item() * accum_steps:.6f} \t TF_Ratio: {teacher_forcing_ratio:.2f} \t {current - start:.2f}s")

        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data_x, data_y) in enumerate(test_loader):
                data_x, data_y = data_x.to(device), data_y.to(device)
                with autocast():
                    # 测试阶段，teacher forcing ratio 绝对必须是 0！纯自己推演！
                    output = model(data_x, future_steps=10, teacher_forcing_ratio=0.0)
                    loss = criterion(output, data_y)
                val_loss += loss.item()
                
                if batch_idx == 0:
                    save_visualization(output.cpu(), data_y.cpu(), epoch, batch_idx)

        avg_val_loss = val_loss / len(test_loader)
        print(f"==== Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} ====")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_convlstm_improved.pth")
            print("Saved new best model!")

if __name__ == '__main__':
    start = time.time()
    train()
