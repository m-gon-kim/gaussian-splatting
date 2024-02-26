from replica_dataset import ReplicaDataset
from tum_dataset import TumDataset
from scannet_dataset import ScannetDataset

class Dataset:
    def __init__(self):
        self.setting = "./setting.txt"

        self.selected_dataset = ""
        self.path_dict = {}
        self.ReadSetting()

        self.dataset_dict = {
            "TUM": TumDataset(),
            "REPLICA": ReplicaDataset(),
            "SCANNET": ScannetDataset()
        }

    def ReadSetting(self):
        setting = open(self.setting)
        for line in setting:
            if line == '\n':
                continue
            key, value = line.strip().split(' = ')
            if key.strip() == 'selected_dataset':
                self.selected_dataset += value.strip()
            elif key.strip() == 'dataset_type':
                dataset_type = value.strip()
                # self.path_dict[dataset_type] = None
            elif key.strip() == 'path':
                path = value.strip()
                self.path_dict[dataset_type] = path

    def GetDataset(self):
        dataset = self.dataset_dict.get(self.selected_dataset, TumDataset())
        dataset.path = self.path_dict.get(self.selected_dataset,
                                          'Z:/TeamFolder/GS_SLAM/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/')

        return dataset

    def InitializeDataset(self):
        dataset = self.GetDataset()
        dataset.InitializeDataset()
