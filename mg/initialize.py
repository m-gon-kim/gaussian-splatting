import math
from replica_dataset import ReplicaDataset
from tum_dataset import TumDataset
from scannet_dataset import ScannetDataset


def GetDataset(dataset_type):
    dataset_dict = {
        "TUM": TumDataset(),
        "REPLICA": ReplicaDataset(),
        "SCANNET": ScannetDataset()
    }
    dataset = dataset_dict.get(dataset_type)
    return dataset


if __name__ == '__main__':
    # EXAMPLE: TUM, REPLICA, SCANNET
    dataset_type = "TUM"
    dataset = GetDataset(dataset_type)
    print(dataset.get_data_len())
    dataset.InitializeDataset()
