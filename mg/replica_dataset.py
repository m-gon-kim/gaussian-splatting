import cv2
import os
import numpy as np
import torch

class ReplicaDataset():
    def __init__(self):
        self.path = ""
        self.novel_view_path = ""
        self.relative_poses = []
        self.nv_relative_poses = []
        self.nv_rgb_list = []
        # self.path = "C:/mg/dataset/Replica/Replica_Dataset/office_0/Sequence_1/"

        # self.img_pair = []
        # self.rgb_list = []
        # self.gray_list = []
        # self.d_list = []

    def get_rgb_list(self):
        rgb_folder = f'{self.path}rgb/'
        rgb_files = [rgb_folder + file for file in os.listdir(rgb_folder)]
        rgb_files = sorted(rgb_files, key=lambda x: int(x.split('rgb_')[-1].split('.')[0]))
        return rgb_files

    def get_nv_rgb_list(self):
        rgb_folder = f'{self.novel_view_path}rgb/'
        rgb_files = [rgb_folder + file for file in os.listdir(rgb_folder)]
        rgb_files = sorted(rgb_files, key=lambda x: int(x.split('rgb_')[-1].split('.')[0]))
        return rgb_files

    def get_depth_list(self):
        depth_folder = f'{self.path}depth/'
        depth_files = [depth_folder + file for file in os.listdir(depth_folder)]
        depth_files = sorted(depth_files, key=lambda x: int(x.split('depth_')[-1].split('.')[0]))
        return depth_files

    def get_data_len(self):
        rgb_list = self.get_rgb_list()
        depth_list = self.get_depth_list()
        assert len(rgb_list) == len(depth_list), "Number of files in depth and RGB folders must be the same"
        data_len = len(rgb_list)
        return data_len

    def read_matrices(self, path):
        matrices_path = f'{path}traj_w_c.txt'
        matrices = []
        with open(matrices_path, 'r') as file:
            for line in file:
                matrix = np.array([np.float32(x) for x in line.split()])
                matrices.append(matrix.reshape(4, 4))
        return matrices

    def get_relative_poses(self):  # np.array
        matrices = self.read_matrices(self.path)
        relative_poses = [np.identity(4, dtype=np.float32)]
        for i in range(1, len(matrices)):
            relative_pose = np.linalg.inv(matrices[0]) @ matrices[i]  # Relative pose calculation
            relative_poses.append(relative_pose)
        return relative_poses

    def get_nv_relative_poses(self):  # torch
        matrices1 = self.read_matrices(self.path)
        matrices2 = self.read_matrices(self.novel_view_path)
        relative_poses = []
        for i in range(0, len(matrices2)):
            relative_pose = np.linalg.inv(matrices1[0]) @ matrices2[i]
            relative_poses.append(relative_pose)
        return relative_poses

    def get_novel_view(self, path):
        self.novel_view_path = path
        self.nv_relative_poses = self.get_nv_relative_poses()
        self.nv_rgb_list = self.get_nv_rgb_list()

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
            d_16bit = cv2.imread(depth_list[cntr], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.0
            d_32bit = d_16bit.astype(np.float32)
            cv2.imwrite(f'{self.path}pair/depth/{str(cntr + 1).zfill(5)}.tiff', d_32bit)

            # self.img_pair.append((rgb, d))
            # self.rgb_list.append(rgb)
            # self.gray_list.append(gray)
            # self.d_list.append(d)

    def get_camera_intrinsic(self):
        # https://github.com/Harry-Zhi/semantic_nerf/issues/41
        return [320.0, 320.0, 319.5, 239.5]

    def ReturnData(self, index):
        file_name = f'{str(index).zfill(5)}.png'
        d_file_name = f'{str(index).zfill(5)}.tiff'
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{d_file_name}', cv2.IMREAD_UNCHANGED)
        pose = self.relative_poses[index - 1]
        return rgb, gray, d, pose
