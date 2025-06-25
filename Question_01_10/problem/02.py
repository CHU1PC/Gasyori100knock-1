import os
import cv2  # cv2は画像処理系でよく使う
import numpy as np

Que1_path = os.path.dirname(os.path.dirname(__file__))


img = cv2.imread(os.path.join(Que1_path, "imori.jpg")).astype(np.float32)

print(img.shape)
grayimg = (img[:, :, 0] * 0.2126 +
           img[:, :, 1] * 0.7152 +
           img[:, :, 2] * 0.0722).astype(np.uint8)


cv2.imshow("gray", grayimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
