import cv2
import numpy as np
import os
from data_processor import DataProcessor
from optical_flow_predictor import OpticalFlowPredictor

def evaluate_mse(pred, truth):
    return np.mean((pred - truth) ** 2)

def main():
    print("Initializing Data Processor...")
    dp = DataProcessor('test_data/1.png')
    predictor = OpticalFlowPredictor()

    # Load 1 to 5
    print("Loading observation sequences...")
    obs_frames = []
    for i in range(1, 14):
        path = f"test_data/{i}.png"
        intensity = dp.image_to_intensity(path)
        obs_frames.append(intensity)
        
    # Load 6 to 10 for ground truth
    print("Loading ground truth sequences...")
    gt_frames = []
    for i in range(15, 18):
        path = f"test_data/{i}.png"
        gt_frames.append(dp.image_to_intensity(path))
        
    # Compute flow between 4 and 5
    print("Computing Optical Flow (frame 4 -> 5)...")
    flow = predictor.calculate_flow(obs_frames[3], obs_frames[4])
    
    # Predict 5 steps into the future
    print("Predicting future 5 steps...")
    preds = predictor.extrapolate(obs_frames[4], flow, steps=5)
    
    # Evaluate and save
    os.makedirs('output', exist_ok=True)
    
    for i, (pred, truth) in enumerate(zip(preds, gt_frames)):
        frame_idx = i + 14
        mse = evaluate_mse(pred, truth)
        print(f"Frame {frame_idx} MSE: {mse:.2f}")
        
        # Visualize
        pred_img = dp.intensity_to_image(pred)
        truth_img = dp.intensity_to_image(truth)
        
        # 为了对比，我们将两张图拼接到一起: 左侧 GT, 右侧 Pred
        combined = np.hstack((truth_img, pred_img))
        
        # 添加文字标注
        cv2.putText(combined, f"Frame {frame_idx} - Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(combined, f"Frame {frame_idx} - Prediction", (truth_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out_path = f"output/compare_{frame_idx}.png"
        cv2.imwrite(out_path, combined)
        print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    main()
