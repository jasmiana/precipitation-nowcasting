import cv2
import numpy as np

class OpticalFlowPredictor:
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0):
        # Farneback 参数
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def calculate_flow(self, prev_frame, next_frame):
        """
        计算稠密光流 (Farneback)
        prev_frame, next_frame: (H, W) 单通道 intensity 数据
        返回: (H, W, 2) 的光流矩阵 [dx, dy]
        """
        # OpenCV 的光流通常以 uint8 为输入，我们的强度是 0-119，可以使用 uint8。
        # 但为了保留细节，也可对 [0, 119] 归一化到 [0, 255]
        prev_gray = np.uint8(prev_frame * (255.0 / 119.0))
        next_gray = np.uint8(next_frame * (255.0 / 119.0))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, self.flags
        )
        return flow

    def extrapolate(self, last_frame, flow, steps):
        """
        采用前向预测 (Semi-Lagrangian advection)
        last_frame: 最新的一帧 intensity (H, W)
        flow: 我们获取到的运动流场 (H, W, 2)
        steps: 要往未来外推几步
        返回: 具有 steps 长度的 list，包含未来帧 intensity
        """
        h, w = last_frame.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        predictions = []
        current_frame = last_frame.astype(np.float32)
        
        # 光流方向为：从 prev 到 next，那么物体的速度是 flow。
        # 为了生成由于这个速度运动而来的下一张图，我们用目标位置去反找原图像素：
        # next_img(x, y) = prev_img(x - dx, y - dy) 
        # 因为我们假设流场随时间不变，所以我们在每一时刻用当前预测图和初始流场外推。
        
        # 因为流场伴随气团移动（理想情况），这里做一个简化：使用固定的欧拉场，即流场空间分布不变。
        map_x = (x - flow[..., 0]).astype(np.float32)
        map_y = (y - flow[..., 1]).astype(np.float32)

        for _ in range(steps):
            # remap 默认双线性插值
            next_pred = cv2.remap(current_frame, map_x, map_y, 
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=0)
            predictions.append(next_pred)
            current_frame = next_pred
            
        return predictions
