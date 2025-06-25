import os
import cv2
import matplotlib.pyplot as plt  # noqa
import numpy as np
assets_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "assets"
    )

img = cv2.imread(os.path.join(assets_path, 'imori.jpg'))
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
# 128 x 128だから一次元にすると256個の要素になる

max_Sb2 = 0.0
max_t = 0
for t in range(0, 256):
    class0 = gray < t
    class1 = gray >= t
    w0 = class0.sum() / class0.size
    w1 = class1.sum() / class1.size
    if w0 == 0 or w1 == 0:
        continue
    M0 = gray[class0].mean()
    M1 = gray[class1].mean()
    Sbt = w0 * w1 * (M0 - M1)**2
    if Sbt > max_Sb2:
        max_Sb2 = Sbt
        max_t = t


binary = np.where(gray < max_t, 0, 255).astype(np.uint8)

cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.hist(gray.ravel(), bins=256, rwidth=0.8, range=(0, 255))
# plt.xlabel('value')
# plt.ylabel('appearance')
# plt.show()
