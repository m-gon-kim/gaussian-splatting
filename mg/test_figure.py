
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import torch

import numpy as np
import cv2



path_con = "C:/lab/research/eccv_fig/concept/contour_img.png"
contour = cv2.imread(path_con)

path_depth = "C:/lab/research/eccv_fig/concept/depth.png"
depth = cv2.imread(path_depth)

updated_contour = contour.copy()
updated_d = depth.copy()

print(updated_contour.shape)

mask = np.all(updated_contour == [100, 255, 100], axis=-1)


# Replace pixels in the image where the mask is True with the new RGB value (0, 255, 0)
updated_contour[mask] = [255, 0, 0]

mask = np.all(updated_contour == [0, 0, 255], axis=-1)
for i in range(-3, 4, 1):
    print(i)
for v in range(mask.shape[0]):
    for u in range(mask.shape[1]):
        if mask[v, u]:
            for i in range(-3, 4, 1):
                new_v = v+ i
                if new_v < 0 or new_v >= 480:
                    continue
                for j in range(-3, 4, 1):
                    new_u = u + j
                    if new_u < 0 or new_u >= 640:
                        continue
                    updated_contour[new_v, new_u] = [0, 0, 255]
                    updated_d[new_v, new_u] = [0, 0, 255]

cv2.imshow("s", updated_contour)
cv2.waitKey(0)
cv2.imwrite("C:/lab/research/eccv_fig/concept/contour_update_img.png", updated_contour)
cv2.imwrite("C:/lab/research/eccv_fig/concept/contour_update_d.png", updated_d)

