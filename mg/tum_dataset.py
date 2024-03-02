import os
import numpy as np
import cv2
import utility

class TumDataset():
    def __init__(self):
        self.path = ""
        # self.img_pair = []
        # self.rgb_list = []
        # self.gray_list = []
        # self.d_list = []

    def read_file_list(self, filename):
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
        list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
        return dict(list)


    def associate(self, first_list, second_list, offset, max_difference):

        first_keys = list(first_list.keys())
        second_keys = list(second_list.keys())
        potential_matches = [(abs(a - (b + offset)), a, b)
                             for a in first_keys
                             for b in second_keys
                             if abs(a - (b + offset)) < max_difference]
        potential_matches.sort()
        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append((a, b))

        matches.sort()
        pair = {}
        for a, b in matches:
            pair[a] = b
        return pair

    def read_matrices(self):
        poses_path = f'{self.path}associated_gt.txt'
        matrices = []
        with open(poses_path, 'r') as file:
            for line in file:
                line = line.rstrip()
                if line.strip() == '':
                    pass
                pose = list(map(float, line.split(' ')))
                matrix = np.eye(4)
                matrix[:3, :3] = utility.qvec2rotmat(pose[3:])
                matrix[:3, 3] = pose[:3]
                matrices.append(matrix)

        return matrices

    def get_relative_poses(self):
        matrices = self.read_matrices()
        relative_poses = [np.identity(4)]
        for i in range(1, len(matrices)):
            relative_pose = np.linalg.inv(matrices[0]) @ matrices[i]  # Relative pose calculation
            relative_poses.append(relative_pose)
        return relative_poses

    def InitializeDataset(self):
        first_list = self.read_file_list(f'{self.path}rgb.txt')
        second_list = self.read_file_list(f'{self.path}depth.txt')
        third_list = self.read_file_list(f'{self.path}groundtruth.txt')

        matches = self.associate(first_list, second_list, 0.0, 0.02)
        gt_matches = self.associate(matches, third_list, 0.0, 0.02)

        pair = []
        for a in matches:
            pair.append((first_list[a][0], second_list[matches[a]][0]))

        cntr = 0

        os.makedirs(f'{self.path}pair/rgb', exist_ok=True)
        os.makedirs(f'{self.path}pair/gray', exist_ok=True)
        os.makedirs(f'{self.path}pair/depth', exist_ok=True)

        for a in pair:
            cntr += 1
            print(cntr)
            rgb = cv2.imread(f'{self.path}{a[0]}')
            cv2.imwrite(f'{self.path}pair/rgb/{str(cntr).zfill(5)}.png', rgb)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            gray = cv2.imread(f'{self.path}{a[0]}', cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'{self.path}pair/gray/{str(cntr).zfill(5)}.png', gray)
            d_16bit = cv2.imread(f'{self.path}{a[1]}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 5000.0
            d_32bit = d_16bit.astype(np.float32)
            cv2.imwrite(f'{self.path}pair/depth/{str(cntr).zfill(5)}.tiff', d_32bit)

        # make poses file (tx ty tz qx qy qz qw)
        pose_file = f'{self.path}associated_gt.txt'
        gt = []
        for a in gt_matches:
            gt.append(third_list[gt_matches[a]])
        with open(pose_file, 'w+') as file:
            for a in gt:
                for b in a:
                    file.write(f'{b} ')
                file.write('\n')
            # self.img_pair.append((rgb, d))
            # self.rgb_list.append(rgb)
            # self.gray_list.append(gray)
            # self.d_list.append(d)

    def get_data_len(self):
        first_list = self.read_file_list(f'{self.path}rgb.txt')
        second_list = self.read_file_list(f'{self.path}depth.txt')
        matches = self.associate(first_list, second_list, 0.0, 0.02)
        data_len = len(matches)
        return data_len

    def get_camera_intrinsic(self):
        # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
        idx = self.path.find("freiburg")
        dataset_name = self.path[idx:idx+9]

        k_dict = {
            "freiburg1": [517.3, 516.5, 318.6, 255.3],
            "freiburg2": [520.9, 521.0,	325.1, 249.7],
            "freiburg3": [535.4, 539.2, 320.1, 247.6]
        }
        fx, fy, cx, cy = k_dict.get(dataset_name, [535.4, 539.2, 320.1, 247.6])

        return [fx, fy, cx, cy]

    def ReturnData(self, index):
        file_name = f'{str(index).zfill(5)}.png'
        d_file_name = f'{str(index).zfill(5)}.tiff'
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{d_file_name}', cv2.IMREAD_UNCHANGED)
        pose = self.get_relative_poses()
        return rgb, gray, d, pose
