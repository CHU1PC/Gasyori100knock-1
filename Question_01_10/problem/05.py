import os
import cv2
import numpy as np

assets_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "assets"
    )

# RGB
img = cv2.imread(os.path.join(assets_path, 'imori.jpg'))

B = img[..., 0].copy() / 255
G = img[..., 1].copy() / 255
R = img[..., 2].copy() / 255

Max = np.max(img, axis=2) / 255
Min = np.min(img, axis=2) / 255

condition = [
    Min == Max,
    Min == B,
    Min == R,
    Min == G
]

h_select = [
    0,
    60 * (G - R) / (Max - Min) + 60,
    60 * (B - G) / (Max - Min) + 180,
    60 * (R - B) / (Max - Min) + 300
]
V = Max
S = Max - Min

H = np.select(condition, h_select)
H = (H + 180) % 360

C = S
H_ = H / 60
X = C * (1 - np.abs(np.mod(H_, 2) - 1))

zeros = np.zeros_like(H_)
condlist = [
    (0 <= H_) & (H_ < 1),
    (1 <= H_) & (H_ < 2),
    (2 <= H_) & (H_ < 3),
    (3 <= H_) & (H_ < 4),
    (4 <= H_) & (H_ < 5),
    (5 <= H_) & (H_ < 6)
]

r_list = [C, X, zeros, zeros, X, C]
g_list = [X, C, C, X, zeros, zeros]
b_list = [zeros, zeros, X, C, C, X]

R1 = np.select(condlist, r_list, default=zeros)
G1 = np.select(condlist, g_list, default=zeros)
B1 = np.select(condlist, b_list, default=zeros)


R = R1 + (V - C)
G = G1 + (V - C)
B = B1 + (V - C)
mask_gray = (Max == Min)
B[mask_gray] = 0
G[mask_gray] = 0
R[mask_gray] = 0

bgr = np.stack([B, G, R], axis=2) * 255
bgr = np.clip(bgr, 0, 255)

cv2.imshow("bgr", bgr.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
