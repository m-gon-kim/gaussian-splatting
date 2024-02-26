from replica_dataset import ReplicaDataset
from tum_dataset import TumDataset
from scannet_dataset import ScannetDataset

class Dataset:
    def __init__(self):
        self.setting = "./setting.txt"
        self.dataset_dict = {
            "TUM": TumDataset(),
            "REPLICA": ReplicaDataset(),
            "SCANNET": ScannetDataset()
        }


    def GetDataset(self):
        setting = open(self.setting)
        lines = setting.readlines()

        keys_dict = {
            "m_colorWidth": self.rgb_width,
            "m_colorHeight": self.rgb_height,
            "m_depthWidth": self.d_width,
            "m_depthHeight": self.d_height,
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