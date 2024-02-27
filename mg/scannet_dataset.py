import cv2
import os
import numpy as np

class ScannetDataset:
    def __init__(self):
        self.path = ""

        # self.img_pair = []
        # self.rgb_list = []
        # self.gray_list = []
        # self.d_list = []

        self.rgb_width = []
        self.rgb_height = []
        self.d_width = []
        self.d_height = []
        self.rgb_intrinsic = []
        self.d_intrinsic = []

        self.info_read = False

    def read_info_file(self, filename):
        with open(filename) as file:
            keys_dict = {
                "m_colorWidth": self.rgb_width,
                "m_colorHeight": self.rgb_height,
                "m_depthWidth": self.d_width,
                "m_depthHeight": self.d_height,
                "m_calibrationColorIntrinsic": self.rgb_intrinsic,
                "m_calibrationDepthIntrinsic": self.d_intrinsic
            }

            for line in file:
                if "=" in line:
                    key, value = line.strip().split(' = ')
                    if key in keys_dict:
                        values = map(float if '.' in value else int, value.split())
                        keys_dict[key].extend(values)

    def initialize_info(self):
        if not self.info_read:
            self.read_info_file(f'{self.path}_info.txt')
            self.info_read = True

    def get_file_list(self, extension):
        return [os.path.join(self.path, filename) for filename in os.listdir(self.path) if filename.endswith(extension)]

    def get_camera_intrinsic(self):
        # depth camera intrinsic
        fx = self.d_intrinsic[0]
        fy = self.d_intrinsic[5]
        cx = self.d_intrinsic[2]
        cy = self.d_intrinsic[6]
        return [fx, fy, cx, cy]

    def get_data_len(self):
        rgb_list = self.get_file_list(".color.jpg")
        depth_list = self.get_file_list(".depth.pgm")
        assert len(rgb_list) == len(depth_list), "Number of files in depth and RGB folders must be the same"
        return len(rgb_list)

    def InitializeDataset(self):
        self.read_info_file(f'{self.path}_info.txt')
        rgb_list = self.get_file_list(".color.jpg")
        depth_list = self.get_file_list(".depth.pgm")
        assert len(rgb_list) == len(depth_list), "Number of files in depth and RGB folders must be the same"

        frames = len(rgb_list)

        os.makedirs(f'{self.path}pair/rgb', exist_ok=True)
        os.makedirs(f'{self.path}pair/gray', exist_ok=True)
        os.makedirs(f'{self.path}pair/depth', exist_ok=True)

        for cntr in range(frames):
            print(cntr)
            rgb = cv2.imread(rgb_list[cntr])
            rgb_resized = cv2.resize(rgb, (self.d_width[0], self.d_height[0]))
            cv2.imwrite(f'{self.path}pair/rgb/{str(cntr + 1).zfill(5)}.png', rgb_resized)

            gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(f'{self.path}pair/gray/{str(cntr + 1).zfill(5)}.png', gray)

            d_16bit = cv2.imread(depth_list[cntr], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.0
            d_32bit = d_16bit.astype(np.float32)
            cv2.imwrite(f'{self.path}pair/depth/{str(cntr + 1).zfill(5)}.tiff', d_32bit)

            # self.img_pair.append((rgb, d))
            # self.rgb_list.append(rgb)
            # self.gray_list.append(gray)
            # self.d_list.append(d)

    def ReturnData(self, index):
        file_name = f'{str(index).zfill(5)}.png'
        d_file_name = f'{str(index).zfill(5)}.tiff'
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{d_file_name}', cv2.IMREAD_UNCHANGED)

        return rgb, gray, d
