import torch
import torch.nn as nn
import random

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # 核心修改：在通道数大幅增加的情况下，依然保持原分辨率
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class DeepSeq2SeqConvLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[32, 32, 32], kernel_size=3):
        """
        无显存限制的高保真版本 (High-Fidelity Version)
        直接在原分辨率下运行多层深层 ConvLSTM。
        """
        super(DeepSeq2SeqConvLSTM, self).__init__()
        
        self.num_layers = len(hidden_dims)
        self.cells = nn.ModuleList()
        
        # 构建多层 ConvLSTM 网络
        for i, h_dim in enumerate(hidden_dims):
            # 第一层的输入是图像通道，后续层的输入是上一层的隐藏状态通道
            cur_input_dim = in_channels if i == 0 else hidden_dims[i-1]
            self.cells.append(
                ConvLSTMCell(input_dim=cur_input_dim, 
                             hidden_dim=h_dim, 
                             kernel_size=kernel_size) # 使用 5x5 的大卷积核
            )
            
        # 解码器变得极其简单：只需要用一个 1x1 卷积将高维特征图还原为单通道灰度图
        # 因为我们没有降采样，所以不需要反卷积 (ConvTranspose2d)
        self.output_conv = nn.Conv2d(in_channels=hidden_dims[-1], 
                                     out_channels=in_channels, 
                                     kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, future_steps=10, teacher_forcing_ratio=0.5, target=None):
        B, T_in, C, H, W = x.size()
        
        # 初始化所有层的 hidden states 和 cell states
        states = [cell.init_hidden(B, (H, W)) for cell in self.cells]
        
        # 1. 吸收历史观测阶段 (Encoder)
        for t in range(T_in):
            input_t = x[:, t]
            for i, cell in enumerate(self.cells):
                states[i] = cell(input_t, states[i])
                input_t = states[i][0] # 当前层的 hidden state 作为下一层的 input
                
        # 开始生成阶段
        outputs = []
        
        # 根据最后一次更新的顶层 hidden state，预测未来的第 1 帧
        current_pred = self.sigmoid(self.output_conv(states[-1][0]))
        outputs.append(current_pred)
        
        # 2. 预测接下来的步数 (Decoder / Forecaster)
        for t in range(1, future_steps):
            # Scheduled sampling
            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t-1]
            else:
                decoder_input = current_pred
                
            input_t = decoder_input
            for i, cell in enumerate(self.cells):
                states[i] = cell(input_t, states[i])
                input_t = states[i][0]
                
            # 从顶层特征提取出画面预测
            current_pred = self.sigmoid(self.output_conv(states[-1][0]))
            outputs.append(current_pred)
            
        return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    # 模拟高保真度压力测试
    model = DeepSeq2SeqConvLSTM(hidden_dims=[64, 64, 64], kernel_size=5).cuda()
    
    # Batch Size 设为 2，否则非常容易 OOM (Out Of Memory)
    dummy_x = torch.zeros((2, 10, 1, 64, 64)).cuda()
    dummy_y = torch.zeros((2, 10, 1, 64, 64)).cuda()
    out = model(dummy_x, future_steps=10, teacher_forcing_ratio=0.5, target=dummy_y)
    
    print("Deep Model Forward Pass Success.")
    print("Output Shape:", out.shape)
    
    # 观察显存消耗（将会比之前大好几倍）
    print("Max memory allocated: {:.2f} MB".format(torch.cuda.max_memory_allocated() / (1024 ** 2)))