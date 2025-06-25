import os
import cv2
import numpy as np

assets_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "assets"
    )

# BGR
img = cv2.imread(os.path.join(assets_path, 'imori.jpg'))

condition = [
    (0 <= img) & (img < 64),
    (64 <= img) & (img < 128),
    (128 <= img) & (img < 192),
    (192 <= img) & (img < 256)
]
choice = [32, 96, 160, 224]

out = np.select(condition, choice, default=0)

cv2.imshow("img", out.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
