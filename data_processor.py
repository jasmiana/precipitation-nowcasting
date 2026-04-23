import cv2
import numpy as np

class DataProcessor:
    def __init__(self, sample_image_path):
        """
        初始化数据处理器，通过样本图提取雷达色板
        """
        self.sample_img = cv2.imread(sample_image_path)
        self.colors = self._extract_colormap()
        self.lut = self._build_lookup_table()
        
    def _extract_colormap(self):
        """
        在 x=1320 (图片最右侧的色板条) 提取梯度颜色。
        """
        col = self.sample_img[:, 1320, :] # BGR
        colors = []
        for y in range(self.sample_img.shape[0]):
            b, g, r = col[y]
            # 过滤白、黑、灰
            if b == g and g == r:
                continue
            color_tuple = (int(r), int(g), int(b))
            if not colors or colors[-1] != color_tuple:
                colors.append(color_tuple)
        return colors

    def _build_lookup_table(self):
        """
        构建 RGB 到 强度 (Intensity) 的 3D 查找表
        共有 N 个颜色阶段，强度 1 到 N。0 代表背景。
        """
        N = len(self.colors)
        # 用 uint8 存即可，因为颜色数目大约在 119 左右
        lut = np.zeros((256, 256, 256), dtype=np.uint8)
        
        for idx, (r, g, b) in enumerate(self.colors):
            # colors 从上到下，索引越小 dBZ 越高，强度应越大。
            # 为了让 0 代表背景，强度由 1 递增至 N
            intensity = N - idx
            lut[r, g, b] = intensity
            
        return lut
        
    def image_to_intensity(self, img_path):
        """
        读取图片，并将 RGB 转换为单通道强度矩阵
        """
        img = cv2.imread(img_path)
        # img 是 BGR 格式，转换为 R, G, B
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        
        # 利用 lookup table 返回二维矩阵
        intensity = self.lut[r, g, b]
        return intensity

    def intensity_to_image(self, intensity):
        """
        将强度矩阵转化回可视化的 BGR 图像
        """
        N = len(self.colors)
        # 生成一个逆向查找表，大小为 N + 1 (包含背景 0)
        # 0 我们映射到纯白背景
        reverse_lut = np.full((N + 1, 3), 255, dtype=np.uint8)
        
        for idx, c in enumerate(self.colors):
            intensity_val = N - idx
            # 注意 OpenCV 储存为 BGR
            r, g, b = c
            reverse_lut[intensity_val] = [b, g, r]
            
        # 考虑到外推预测可能会产生浮点或越界的强度，先截断到 0~N 并转为整型
        clipped = np.clip(np.round(intensity), 0, N).astype(np.uint8)
        # map
        bgr_img = reverse_lut[clipped]
        return bgr_img

if __name__ == "__main__":
    # Test DataProcessor
    dp = DataProcessor('test_data/1.png')
    print(f"Extracted {len(dp.colors)} unique colors.")
    
    # 验证正向映射
    intensity = dp.image_to_intensity('test_data/1.png')
    print(f"Intensity max: {intensity.max()}, min: {intensity.min()}")
    
    # 验证逆向渲染
    restored = dp.intensity_to_image(intensity)
    cv2.imwrite('restored_1.png', restored)
    print("Saved restored image to restored_1.png")
