import cv2
import numpy as np

img = cv2.imread('test_data/1.png')
col = img[:, 1320, :] # BGR
y_indices = []
colors = []

for y in range(img.shape[0]):
    b, g, r = col[y]
    # ignoring white, black, gray
    if (b == g and g == r):
        continue
    # Add to list if it's a new color or same color block
    color_tuple = (int(r), int(g), int(b))
    if not colors or colors[-1] != color_tuple:
        colors.append(color_tuple)
        y_indices.append(y)

print(f"Found {len(colors)} distinct color blocks/pixels.")
for i, c in enumerate(colors):
    print(f"Color {i}: {c} at start y={y_indices[i]}")
