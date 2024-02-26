import cv2
import os

class ScannetDataset:
    def __init__(self):
        self.path = "Z:/TeamFolder/GS_SLAM/ScanNet/scans/scene0059_00/out/"

        self.img_pair = []
        self.rgb_list = []
        self.gray_list = []
        self.d_list = []

        self.rgb_width = []
        self.rgb_height = []
        self.d_width = []
        self.d_height = []
        self.d_shift = []
        self.rgb_intrinsic = []
        self.d_intrinsic = []

        self.read_info_file(f'{self.path}_info.txt')

    def read_info_file(self, filename):
        file = open(filename)
        lines = file.readlines()
        keys_dict = {
            "m_colorWidth": self.rgb_width,
            "m_colorHeight": self.rgb_height,
            "m_depthWidth": self.d_width,
            "m_depthHeight": self.d_height,
            "m_depthShift": self.d_shift,
            "m_calibrationColorIntrinsic": self.rgb_intrinsic,
            "m_calibrationDepthIntrinsic": self.d_intrinsic
        }

        for line in lines:
            key, value = line.strip().split(' = ')
            if key in keys_dict:
                if '.' in value:
                    value_list = map(float, value.split())
                else:
                    value_list = map(int, value.split())
                keys_dict[key] += value_list

    def get_rgb_list(self):
        rgb_files = []
        for filename in os.listdir(self.path):
            if filename.endswith(".color.jpg"):
                rgb_files += [self.path + filename]
        return rgb_files

    def get_depth_list(self):
        depth_files = []
        for filename in os.listdir(self.path):
            if filename.endswith(".depth.pgm"):
                depth_files += [self.path + filename]
        return depth_files

    def get_camera_intrinsic(self):
        # depth camera intrinsic
        fx = self.d_intrinsic[0]
        fy = self.d_intrinsic[5]
        cx = self.d_intrinsic[2]
        cy = self.d_intrinsic[6]
        return [fx, fy, cx, cy]

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
            rgb_resized = cv2.resize(rgb, (self.d_width[0], self.d_height[0]))
            cv2.imwrite(f'{self.path}pair/rgb/{str(cntr + 1).zfill(5)}.png', rgb_resized)

            gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(f'{self.path}pair/gray/{str(cntr + 1).zfill(5)}.png', gray)

            d = cv2.imread(depth_list[cntr], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / self.d_shift[0]
            cv2.imwrite(f'{self.path}pair/depth/{str(cntr + 1).zfill(5)}.png', d)

            # self.img_pair.append((rgb, d))
            # self.rgb_list.append(rgb)
            # self.gray_list.append(gray)
            # self.d_list.append(d)

    def ReturnData(self, index):
        file_name = f'{str(index).zfill(5)}.png'
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{file_name}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        return rgb, gray, d
