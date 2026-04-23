import cv2
import numpy as np

img = cv2.imread('test_data/1.png')
h, w, c = img.shape
print(f"Image shape: {h}x{w}x{c}")

# Let's sample a few rows halfway down the image, from right to left
# to find the color bar
y = h // 2
for x in range(w-1, w-100, -1):
    print(f"Pixel at x={x}: {img[y, x]}")

