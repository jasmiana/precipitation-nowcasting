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

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
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


class Seq2SeqConvLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        """
        为了 8GB 显存设计的微型序列到序列架构。
        输入 (B, T, C, H, W) -> CNN 降采样 -> ConvLSTM -> CNN 解码 -> 输出 (B, 1, H, W)
        默认用在 64x64 的 MovingMNIST 上。
        """
        super(Seq2SeqConvLSTM, self).__init__()
        
        # Encoder: 1 channel -> 16 -> 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 核心时序模块，在缩小后的特征图 16x16 上运行，极大节省显存
        self.convlstm = ConvLSTMCell(input_dim=32, hidden_dim=hidden_dim, kernel_size=3)
        
        # Decoder: 把时序传递出的特征上采样回原样
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 16, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid()  # 因为 MovingMNIST 归一化到了 [0, 1]
        )
        
    def forward(self, x, future_steps=10, teacher_forcing_ratio=0.5, target=None):
        """
        x: (B, T_in, C, H, W)
        target: (B, T_out, C, H, W) 用于 scheduled sampling
        """
        B, T_in, C, H, W = x.size()
        
        # 编码器输出的尺寸应该是原图的 1/4 (由于两层 stride=2)
        h, w = H // 4, W // 4
        
        # 初始化 ConvLSTM 的 hidden states
        hidden_state = self.convlstm.init_hidden(batch_size=B, image_size=(h, w))
        
        # 1. 吸收历史观测阶段 (Teacher Forcing always True for past sequences)
        for t in range(T_in):
            input_xt = self.encoder(x[:, t])
            hidden_state = self.convlstm(input_xt, hidden_state)
            
        # 开始生成阶段
        outputs = []
        # 我们用历史的最后一帧的预测结果，作为解码和下一步的基石开始
        # 但第一步的推演是基于过去的 hidden_state 提取出来的
        h_t, c_t = hidden_state
        decoder_output = self.decoder(h_t) # 预测出未来的第 1 帧
        outputs.append(decoder_output)
        
        # 2. 预测接下来的步数
        current_pred = decoder_output
        for t in range(1, future_steps):
            # Scheduled sampling 机制:
            if target is not None and random.random() < teacher_forcing_ratio:
                # 使用 Ground Truth
                decoder_input = target[:, t-1]
            else:
                # 使用上一步的自身的生成结果
                decoder_input = current_pred
                
            enc_in = self.encoder(decoder_input)
            hidden_state = self.convlstm(enc_in, hidden_state)
            
            h_t, c_t = hidden_state
            current_pred = self.decoder(h_t)
            outputs.append(current_pred)
            
        # 将 outputs stacking 成 (B, T_out, C, H, W)
        return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    # 模拟显存空转压力测试
    model = Seq2SeqConvLSTM().cuda()
    dummy_x = torch.zeros((2, 10, 1, 64, 64)).cuda()
    dummy_y = torch.zeros((2, 10, 1, 64, 64)).cuda()
    out = model(dummy_x, future_steps=10, teacher_forcing_ratio=0.5, target=dummy_y)
    
    print("Model Forward Pass Success.")
    print("Output Shape:", out.shape)
    import torch
    print("Max memory allocated:", torch.cuda.max_memory_allocated() / (1024 ** 2), "MB")
