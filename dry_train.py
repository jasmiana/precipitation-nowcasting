import torch
import torch.nn as nn
from model import Seq2SeqConvLSTM
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

def run_dry_test():
    print("Starting Training Loop Dry-Run (Synthetic Data)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqConvLSTM(hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # 构造假数据 (Batch=2, T=20, 1, 64, 64)
    dummy_x = torch.rand((2, 10, 1, 64, 64), device=device)
    dummy_y = torch.rand((2, 10, 1, 64, 64), device=device)
    
    model.train()
    optimizer.zero_grad()
    
    with autocast():
        # 这里测试 Scheduled Sampling 为 0.5
        out = model(dummy_x, target=dummy_y, teacher_forcing_ratio=0.5)
        loss = criterion(out, dummy_y)
        loss = loss / 4 # 测试 accum step
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print("Training Loop Dry-Run Passed! No OOM or Autocast Syntax Errors.")
    print("Final Output Shape:", out.shape)
    
if __name__ == '__main__':
    run_dry_test()
