import cv2
import os
import numpy as np

class ReplicaDataset:
    def __init__(self):
        self.path = "Z:/TeamFolder/GS_SLAM/Replica/Replica_Dataset/room_0/room_0/Sequence_1/"
        # self.path = "C:/mg/dataset/Replica/Replica_Dataset/office_0/Sequence_1/"

        self.img_pair = []
        self.rgb_list = []
        self.gray_list = []
        self.d_list = []

    def get_rgb_list(self):
        rgb_folder = f'{self.path}rgb/'
        rgb_files = [rgb_folder + file for file in os.listdir(rgb_folder)]

        return rgb_files

    def get_depth_list(self):
        depth_folder = f'{self.path}depth/'
        depth_files = [depth_folder + file for file in os.listdir(depth_folder)]

        return depth_files

    def get_data_len(self):
        rgb_list = self.get_rgb_list()
        depth_list = self.get_depth_list()
        assert len(rgb_list) == len(depth_list), "Number of files in depth and RGB folders must be the same"
        data_len = len(rgb_list)
        return data_len

    def InitializeDataset(self):
        rgb_list = self.get_rgb_list()
        depth_list = self.get_depth_list()
        assert len(rgb_list) == len(depth_list), "Number of files in depth and RGB folders must be the same"

        frames = len(rgb_list)

        os.makedirs(f'{self.path}pair/rgb', exist_ok=True)
        os.makedirs(f'{self.path}pair/gray', exist_ok=True)
        os.makedirs(f'{self.path}pair/depth', exist_ok=True)

        for cntr in range(frames):
            print(cntr)
            rgb = cv2.imread(rgb_list[cntr])
            # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{self.path}pair/rgb/{str(cntr + 1).zfill(5)}.png', rgb)
            gray = cv2.imread(rgb_list[cntr], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'{self.path}pair/gray/{str(cntr + 1).zfill(5)}.png', gray)
            d = cv2.imread(depth_list[cntr], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.0
            cv2.imwrite(f'{self.path}pair/depth/{str(cntr + 1).zfill(5)}.png', d)

            # self.img_pair.append((rgb, d))
            # self.rgb_list.append(rgb)
            # self.gray_list.append(gray)
            # self.d_list.append(d)

    def get_camera_intrinsic(self):
        # https://github.com/Harry-Zhi/semantic_nerf/issues/41
        return [320.0, 320.0, 319.5, 239.5]

    def ReturnData(self, index):
        file_name = f'{str(index).zfill(5)}.png'
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{file_name}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        return rgb, gray, d
