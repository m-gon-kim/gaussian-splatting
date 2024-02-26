
import os
import cv2
class TumDataset:
    def __init__(self):
        self.path = "Z:/TeamFolder/GS_SLAM/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/"
        self.img_pair = []
        self.rgb_list = []
        self.gray_list = []
        self.d_list = []

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

    def InitializeDataset(self):
        first_list = self.read_file_list(f'{self.path}rgb.txt')
        second_list = self.read_file_list(f'{self.path}depth.txt')

        matches = self.associate(first_list, second_list, 0.0, 0.02)
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
            d = cv2.imread(f'{self.path}{a[1]}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 5000.0
            cv2.imwrite(f'{self.path}pair/depth/{str(cntr).zfill(5)}.png', d)

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
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{file_name}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return rgb, gray, d