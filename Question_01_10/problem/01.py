import os
import cv2  # cv2は画像処理系でよく使う
import numpy as np

Que1_path = os.path.dirname(os.path.dirname(__file__))


def BGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


# rgb -> bgr
# Read image
# cv2で画像を読み取ると(H, W, channel size)でndarrayとして出てくる
img = cv2.imread(os.path.join(Que1_path, "imori.jpg"))

# BGR -> RGB
img_BGR = np.take(img, [2, 1, 0], axis=2)


# Save result
# cv2.imwrite("out.jpg", img_BGR)
cv2.imshow("result", img_BGR)
# なにかキー入力されたら消す
cv2.waitKey(0)
cv2.destroyAllWindows()
