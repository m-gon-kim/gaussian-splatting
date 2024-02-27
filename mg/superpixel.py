
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import torch

import numpy as np
import cv2


class SuperPixelManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.iteration_N = 1
        self.region_size = 16
        self.ruler = 100
        self.device = "cuda"
        with torch.no_grad():
            self.indices = torch.zeros((2, self.height, self.width), dtype=torch.float32)
        self.SetIndices()

    def SetIndices(self):
        for i in range(self.height):
            for j in range(self.width):
                self.indices[0, i, j] = i
                self.indices[1, i, j] = j

    def ComputeSuperPixel(self, rgb):
        slic = cv2.ximgproc.createSuperpixelSLIC(rgb, algorithm=102, region_size=self.region_size, ruler=self.ruler)
        slic.iterate(self.iteration_N)

        num_torch = torch.from_numpy(slic.getLabels())
        num_slic = slic.getNumberOfSuperpixels()
        index_list = []

        with torch.no_grad():
            for cls_lbl in range(num_slic):
                mask = num_torch[:, :].eq(cls_lbl)
                masked_indicies = self.indices[:, mask]
                y = int(torch.mean(masked_indicies[0, :]))
                x = int(torch.mean(masked_indicies[1, :]))
                index_list.append((y, x))
            result = torch.tensor(index_list).to(self.device).T
        return result
