import os
import cv2
import numpy as np

img = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "imori.jpg")).astype(np.float32)

img = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722)
###############################
# img[img < 128] = 0
# img[img >= 128] = 255

###############################
img = np.where(img < 128, 0, 255)
img = img.astype(np.uint8)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
