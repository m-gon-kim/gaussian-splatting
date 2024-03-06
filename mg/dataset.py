from replica_dataset import ReplicaDataset
from tum_dataset import TumDataset
from scannet_dataset import ScannetDataset
from replica_dataset2 import ReplicaDataset2
import yaml

class Dataset:
    def __init__(self):
        self.setting = "./setting.txt"
        self.default_dataset_path = 'Z:/TeamFolder/GS_SLAM/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/'

        self.selected_dataset = ""
        self.nv_path = ""
        self.path_dict = {}
        self.ReadSetting()

        self.dataset_dict = {
            "TUM": TumDataset(),
            "REPLICA": ReplicaDataset(),
            "SCANNET": ScannetDataset(),
            "REPLICA2": ReplicaDataset2()
        }
        self.parameters = {}
        self.ReadParameters()

    def ReadSetting(self):
        with open(self.setting) as setting_file:
            dataset_type = ""
            for line in setting_file:
                if line.strip():
                    key, value = line.strip().split(' = ')
                    key = key.strip()
                    value = value.strip()

                    if key == 'selected_dataset':
                        self.selected_dataset = value
                    elif key.strip() == 'dataset_type':
                        dataset_type = value
                    elif key.strip() == 'path':
                        self.path_dict[dataset_type] = value
                    elif key.strip() == 'novel_view_path':
                        self.nv_path = value

    def GetDataset(self):
        dataset = self.dataset_dict.get(self.selected_dataset, TumDataset())
        dataset.path = self.path_dict.get(self.selected_dataset, self.default_dataset_path)
        if isinstance(dataset, ScannetDataset):
            dataset.initialize_info()
        elif isinstance(dataset, ReplicaDataset):
            dataset.get_novel_view(self.nv_path)
        dataset.relative_poses = dataset.get_relative_poses()
        return dataset

    def ReadParameters(self):
        path = "parameters/" + (self.selected_dataset).lower() + "/parameter.yml"
        with open(path, 'r', encoding='utf-8') as file:
            self.parameters = yaml.safe_load(file)


    def InitializeDataset(self):
        dataset = self.dataset_dict.get(self.selected_dataset, TumDataset())
        dataset.path = self.path_dict.get(self.selected_dataset, self.default_dataset_path)
        dataset.InitializeDataset()
